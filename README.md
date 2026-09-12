# Airline-Flight-Delay-Prediction-Analysis
A data-driven study of the T_ONTIME_REPORTING flight dataset (544,003 flights) to uncover what drives arrival delays and to build a classification model that predicts whether an operating flight will land 15+ minutes behind schedule.
# ✈️ Airline Flight Delay Prediction & Analysis

A data science project that cleans, explores, and models one month of U.S. domestic flight data to
predict whether an operating flight will **arrive 15 or more minutes late** — and to explain *why*
delays happen in the first place. The final model is deployed as a live Streamlit app, and the
findings are also presented through an interactive Power BI dashboard.

**Track:** Advanced Data Analysis · **Initiative:** NTI · **Supervisor:** Eng. Huda Hemdan

**Team:** Mohand Yasser · Abdelrahman Ashraf · Bassel Deghady · Joseph Hossam

**Live app:** [flight-delay-prediction-nti.streamlit.app](https://flight-delay-prediction-nti.streamlit.app/)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Phases](#project-phases)
4. [Phase 3 — Data Cleaning](#phase-3--data-cleaning)
5. [Phase 5 — Modeling](#phase-5--modeling)
6. [Phase 6 — Evaluation](#phase-6--evaluation)
7. [Phase 7 — Deployment](#phase-7--deployment)
8. [Power BI Dashboard](#power-bi-dashboard)
9. [How to Run Locally](#how-to-run-locally)
10. [Key Limitations](#key-limitations)

---

## Problem Statement

Flight delays create real operational and customer-service costs for airlines and airports. This
project uses historical flight and operational data to understand *what drives delays* and to
predict, ahead of time, whether an operating flight will arrive 15 or more minutes late.

- **Target column:** `Arrival Delay ≥ 15 Minutes` (binary — `0` = on-time, `1` = delayed 15+ minutes)
- **Problem type:** Binary classification
- **Prediction scenario:** Predict delay risk using only information available around or before
  departure — not information that is only known after or during the flight (to avoid data leakage)
- **Success criterion:** F1-score ≥ 0.70 on the held-out test set, with particular attention to
  Recall for the delayed class

## Dataset

| Item | Value |
|---|---|
| Source file | `T_ONTIME_REPORTING.xlsx` |
| Rows | 544,003 |
| Original columns | 24 |
| Time coverage | 1–31 January 2026 |
| Operating carriers | 13 |
| Origin / destination airports | 341 / 342 |

## Project Phases

| Phase | Description |
|---|---|
| 1–2 | Business Understanding & Data Understanding |
| 3 | Data Cleaning |
| 4 | Exploratory Data Analysis |
| 5 | Modeling |
| 6 | Evaluation |
| 7 | Deployment |

---

## Phase 3 — Data Cleaning

**Output file:** `cleaned_flight_data.csv`

### Initial Data Exploration

Before cleaning, the raw data was profiled with `df.shape`, `df.info()`, `df.describe()`,
`df.duplicated().sum()` (result: **0 duplicates**), and `df.isnull().sum()`.

Missing values found:

| Column | Missing Values |
|---|---|
| `TAIL_NUM` | 4,638 |
| `DEP_DELAY` / `DEP_DEL15` | 25,229 |
| `TAXI_OUT` | 25,515 |
| `ARR_DELAY` / `ARR_DEL15` | 26,781 |
| `CANCELLATION_CODE` | 518,368 |
| `ACTUAL_ELAPSED_TIME` | 26,781 |

### Cleaning Steps

1. **`FL_DATE`** — stripped the redundant time component (`00:00:00`) by converting to `date` and
   back to `datetime`.
2. **`OP_UNIQUE_CARRIER`** — replaced short carrier codes (e.g. `AA`, `DL`, `UA`) with full airline
   names (e.g. American Airlines, Delta Air Lines) using a dictionary mapping.
3. **`TAIL_NUM`** — filled the 4,638 missing values with `"Unknown"`.
4. **`ORIGIN_CITY_NAME`** — kept only the city name, dropping the state (e.g. `"Los Angeles, CA"` →
   `"Los Angeles"`).
5. **`DEST_CITY_NAME`** — same treatment as above, applied to the destination city.
6. **`DEP_DELAY`** — missing values imputed with the **median**, then cast to `int64`.
7. **`DEP_DEL15`** (departure delay ≥ 15 min) — rather than imputing missing values directly, this
   column was fully **recomputed** from the cleaned `DEP_DELAY`: `1` if `DEP_DELAY >= 15`, else `0`.
8. **`ARR_DELAY`** — missing values imputed with the **median**, then cast to `int64`.
9. **`ARR_DEL15`** (arrival delay ≥ 15 min) — same logic as `DEP_DEL15`, recomputed from the cleaned
   `ARR_DELAY`.
10. **`TAXI_OUT`** — missing values imputed with the **mean**.
11. **`CANCELLATION_CODE`** — column dropped entirely (518,368 of 544,003 rows were missing — almost
    no usable signal).
12. **`ACTUAL_ELAPSED_TIME`** — missing values imputed with the **median**, then cast to `int64`.
13. **`CRS_DEP_TIME`** (scheduled departure time) — converted from a raw numeric value (e.g. `1358`)
    into a readable `HH:MM` format via a custom `convert_time()` function.
14. **`CRS_ARR_TIME`** (scheduled arrival time) — same conversion, using the same `convert_time()`
    function.
15. **`ARR_TIME_BLK`** (arrival time block) — converted from `2200-2259` into a readable
    `22:00-22:59` format via a custom `convert_time_block()` function.
16. **Column renaming** for clarity:

    | Old Name | New Name |
    |---|---|
    | `OP_UNIQUE_CARRIER` | Operating Carrier |
    | `TAIL_NUM` | Aircraft Tail Number |
    | `CRS_DEP_TIME` | Scheduled Departure Time |
    | `DEP_DEL15` | Departure Delay ≥ 15 Minutes |
    | `CRS_ARR_TIME` | Scheduled Arrival Time |
    | `ARR_DEL15` | Arrival Delay ≥ 15 Minutes |
    | `ARR_TIME_BLK` | Arrival Time Block |

17. **Data type adjustments:**
    - Converted to **string** (identifiers, not numeric quantities): `OP_CARRIER_FL_NUM`,
      `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID`
    - Converted to **int**: `Departure Delay ≥ 15 Minutes`, `Arrival Delay ≥ 15 Minutes`
18. **Export** — saved the cleaned dataset to `cleaned_flight_data.csv`.

### Result

- **23 columns** after cleaning (24 − `CANCELLATION_CODE`)
- **No missing values** remaining
- **No duplicate rows**
- All time columns are in a readable `HH:MM` format
- Carrier and city names are clear and human-readable

---

## Phase 5 — Modeling

### Data Preparation for Modeling

1. **Removed cancelled flights** (`CANCELLED == 1`, 25,635 rows) — a cancelled flight has no real
   arrival outcome, so its target value would be artificial rather than a genuine observation.
2. **Feature selection** — dropped columns that either cause data leakage or carry no predictive
   meaning:

   | Column | Reason for Removal |
   |---|---|
   | `ARR_DELAY` | Target is derived directly from this column (leakage) |
   | `ACTUAL_ELAPSED_TIME` | Only known after the flight lands |
   | `Arrival Time Block` | Reflects actual arrival, known only after landing |
   | `TAXI_OUT` | Only known after departure |
   | `CANCELLED` | Constant after filtering (always 0) |
   | `Aircraft Tail Number` | High-cardinality identifier, no predictive value |
   | `OP_CARRIER_FL_NUM` | Flight number identifier, not a feature |
   | `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID` | Duplicate of `ORIGIN` / `DEST` |
   | `FL_DATE` | Single month in the data, no usable variation |
   | `Scheduled Arrival Time` | Conceptually tied to the target (arrival), excluded as a precaution |

3. **Feature engineering** — extracted `Departure Hour` (0–23) from `Scheduled Departure Time` to
   capture time-of-day delay patterns.
4. **Encoding** — grouped low-frequency airports into an `"Other"` category (top 20 origins /
   destinations kept), then applied one-hot encoding (`pd.get_dummies`, `drop_first=True`) to
   `Operating Carrier`, `ORIGIN`, and `DEST`.
5. **Train/test split** — 80/20 split with `stratify=y` to preserve the class balance
   (≈80% on-time / 20% delayed) in both sets.
6. **Scaling** — `StandardScaler` fit on the training set only, then applied to both sets.

### Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, interpretable model (`max_iter=2000`) |
| Decision Tree | `max_depth=8` to control overfitting |

---

## Phase 6 — Evaluation

Accuracy alone is not sufficient given the class imbalance (~80/20), so Precision, Recall, and
F1-score were also reported, per the success criteria defined in Phase 1 (**F1 ≥ 0.70**).

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.9219 | 0.8903 | **0.7110** | **0.7906** |
| Decision Tree | 0.9222 | 0.9057 | 0.6974 | 0.7880 |

**Selected model: Logistic Regression** — highest F1-score, and higher Recall for the delayed
class, which matters most operationally (missing an actual delay is more costly than a false
alarm).

- Result **exceeds the Phase 1 target** of F1 ≥ 0.70.
- Confusion matrix, classification report, and feature-importance analysis (via `lr.coef_`) are
  included in the notebook.
- **Top predictive factors:** `DEP_DELAY` (by far the strongest), destination airport (e.g. `ORD`),
  and operating carrier.

---

## Phase 7 — Deployment

A Streamlit app (`app.py`) loads the saved model artifacts and serves live predictions.

**Live demo:** [flight-delay-prediction-nti.streamlit.app](https://flight-delay-prediction-nti.streamlit.app/)

### Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `requirements.txt` | Dependencies |
| `logistic_model.pkl` | Trained Logistic Regression model |
| `scaler.pkl` | Fitted `StandardScaler` |
| `model_columns.pkl` | Exact column order/names used during training |
| `delay_rate_by_carrier.csv` | EDA summary table used in the app's Insights tab |
| `delay_rate_by_hour.csv` | EDA summary table used in the app's Insights tab |

### How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### App Structure

- **Prediction tab:** the user selects carrier, origin, destination, distance, departure hour, and
  current departure delay; the app returns a delayed/on-time prediction with probability.
- **Insights tab:** reuses the Phase 4/5 EDA findings (delay rate by carrier, delay rate by
  departure hour) and displays the final model's evaluation metrics.

### Design Note

Dropdown options in the app are derived directly from `model_columns.pkl` rather than hardcoded,
guaranteeing the input encoding always matches what the model was trained on.

---

## Power BI Dashboard

Alongside the notebook-based analysis, the project's EDA findings are also presented through an
interactive Power BI dashboard with five report pages:

- **Cover** — landing page and navigation hub
- **Flight Overview** — total flights, cancellation rate, on-time rate, and average delays
- **Delay Analysis** — delay by distance range, time of day, and departure/arrival delay
  relationship
- **Airline Performance** — delay rate, cancellation rate, and on-time rate by carrier
- **Route & Airport Analysis** — delay rate by route/state and total flights by airport

---

## Key Limitations

- One category per encoded feature (carrier / origin / destination) is treated as the baseline
  (`drop_first=True`) and is therefore not selectable in the app UI — a minor UX trade-off from
  one-hot encoding.
- Data covers a single month (January 2026); findings should not be generalized beyond this period
  without further validation.
- The dataset does not contain direct weather variables, so weather-related causes of delay cannot
  be modeled directly from the available fields.
