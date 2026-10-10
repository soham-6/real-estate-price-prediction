# Real Estate Price Predictor — Concepts & Working

## Dataset

**California Housing Dataset** (OpenML, version 1) contains ~20,000 block-level records from the 1990 California census. Each row represents a census block group with features like median income, house age, room counts, population, and ocean proximity. The target is `median_house_value` in USD.

---

## Feature Engineering

Three ratio features are derived from the raw columns before training. These are well-known to significantly improve accuracy on this dataset:

| Feature | Formula | Why it helps |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Normalises room count to per-house level |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Captures house type (studio vs family) |
| `population_per_household` | `population / households` | Measures neighbourhood density |

---

## Preprocessing Pipeline

Handled by a **scikit-learn `ColumnTransformer`** inside a `Pipeline`:

- **`SimpleImputer(strategy='median')`** — fills the ~207 missing `total_bedrooms` values (and inherited NaNs in engineered features) with the training-set median. Applied before scaling to avoid errors.
- **`StandardScaler`** — centres each numeric feature to mean=0, std=1. Necessary for Linear Regression so that feature magnitudes don't distort the coefficients.
- **`OneHotEncoder(drop='first')`** — converts `ocean_proximity` (5 categories) into 4 binary columns. `drop='first'` removes one column to avoid multicollinearity.

All transformers are fit **only on training data**, preventing data leakage into the test set.

---

## Models

### 1. Linear Regression
Fits a straight-line relationship: `price = w₁·feature₁ + w₂·feature₂ + … + b`. Fast and interpretable but assumes linearity, so it misses non-linear patterns in housing prices.

### 2. Decision Tree (CART)
Recursively splits the feature space into rectangular regions, each predicting a constant value. `max_depth=10` and `min_samples_leaf=10` control overfitting — too deep and the tree memorises training data.

### 3. Random Forest
An **ensemble** of 200 decision trees, each trained on a random bootstrap sample of the data and a random subset of features (`max_features=0.5`). Final prediction is the average across all trees. Averaging cancels out individual tree errors, giving substantially lower RMSE than a single tree.

---

## Train / Test Split

80% of data is used for training, 20% is held out as an unseen test set (`random_state=42` for reproducibility). All metrics shown in the app are computed on this test set — they reflect real generalisation performance, not training performance.

---

## Evaluation Metrics

| Metric | Meaning | Good when |
|---|---|---|
| **R²** | Fraction of variance explained (0–1) | Closer to 1 |
| **RMSE** | Average error magnitude (penalises large errors) | Lower |
| **MAE** | Average absolute error (more robust to outliers) | Lower |

---

## Currency Conversion

Predicted prices are in USD. INR values are derived by multiplying by a fixed indicative rate (`1 USD = ₹84`). Large amounts are formatted as **Lakhs (L)** or **Crores (Cr)** following the Indian numbering system.

---

## Streamlit & Caching

The app uses `@st.cache_resource` to train all models **once per server session**. Subsequent interactions (slider changes, button clicks) reuse the cached preprocessor and fitted models — no retraining occurs. This keeps the UI responsive.

---

## App Flow

```
User adjusts sliders
        ↓
build_input_df()  →  engineer_features()  →  preprocessor.transform()
        ↓
selected_model.predict()
        ↓
Display USD price  +  INR equivalent
```
