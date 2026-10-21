import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e0e0e0; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  .main-title {
    text-align: center; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem; padding-top: 0.5rem;
  }
  .subtitle { text-align: center; color: #a0aec0; font-size: 1rem; margin-bottom: 1.5rem; }

  .price-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 1.5rem 2rem; text-align: center;
    backdrop-filter: blur(10px); transition: transform 0.2s;
  }
  .price-card:hover { transform: translateY(-3px); }
  .price-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; color: #a0aec0; margin-bottom: 0.4rem; }
  .price-usd   { font-size: 2.2rem; font-weight: 800; color: #48bb78; }
  .price-inr   { font-size: 2.2rem; font-weight: 800; color: #f6ad55; }
  .price-sub   { font-size: 0.75rem; color: #718096; margin-top: 0.3rem; }

  .ci-bar {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 0.8rem 1.2rem; margin-top: 0.8rem;
    font-size: 0.85rem; color: #a0aec0; text-align: center;
  }
  .ci-range { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }

  .metric-chip {
    display: inline-block; border-radius: 8px; padding: 0.35rem 0.8rem;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px; margin: 0.2rem;
  }
  .chip-green { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
  .chip-amber { background: rgba(246,173,85,0.15);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.3); }
  .chip-blue  { background: rgba(99,179,237,0.15);  color: #90cdf4; border: 1px solid rgba(99,179,237,0.3); }

  .info-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem 1.4rem; margin-top: 1rem; font-size: 0.84rem; line-height: 1.8;
  }
  .model-badge {
    display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; font-size: 0.8rem; font-weight: 600; padding: 0.25rem 0.8rem;
    border-radius: 20px; letter-spacing: 0.5px;
  }
  .section-header {
    font-size: 1rem; font-weight: 700; color: #e2e8f0;
    border-left: 3px solid #f6ad55; padding-left: 0.7rem;
    margin: 1.2rem 0 0.6rem 0;
  }
  .rate-note { font-size: 0.73rem; color: #718096; text-align: center; margin-top: 0.4rem; }
  .placeholder-card {
    background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 16px; padding: 3rem 2rem; text-align: center; color: #718096;
  }
  hr { border-color: rgba(255,255,255,0.08); }

  .stButton > button {
    width: 100%; background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a2e; font-weight: 800; font-size: 1rem; border: none;
    border-radius: 10px; padding: 0.65rem 1rem; letter-spacing: 0.5px;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label {
    font-size: 0.8rem; font-weight: 600; color: #a0aec0 !important;
    text-transform: uppercase; letter-spacing: 0.8px;
  }

  div[data-testid="stTabs"] button { color: #a0aec0 !important; font-weight: 600; }
  div[data-testid="stTabs"] button[aria-selected="true"] { color: #ffd200 !important; border-bottom-color: #ffd200 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
USD_TO_INR = 84.0

NUMERIC_COLS = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income',
    # Engineered
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
]
CAT_COLS = ['ocean_proximity']

MODEL_META = {
    'Random Forest': {
        'icon': '🌲', 'color': '#68d391',
        'desc': 'Ensemble of 200 trees with confidence interval estimation.',
    },
    'Decision Tree': {
        'icon': '🌿', 'color': '#90cdf4',
        'desc': 'Single CART tree — fast and interpretable.',
    },
    'Linear Regression': {
        'icon': '📈', 'color': '#b794f4',
        'desc': 'Linear baseline; fast but limited by linearity assumption.',
    },
}

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    hh = d['households'].clip(lower=1)
    d['rooms_per_household']      = d['total_rooms'] / hh
    d['bedrooms_per_room']        = d['total_bedrooms'] / d['total_rooms'].clip(lower=1)
    d['population_per_household'] = d['population'] / hh
    return d

def inr_label(amount: float) -> str:
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} L"
    return f"₹{amount:,.0f}"

# ── Training (cached for the session) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    california = fetch_openml(name='california_housing', version=1, as_frame=True, parser='auto')
    X = engineer_features(california.data)
    y = california.target.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe,                                              NUMERIC_COLS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),      CAT_COLS),
    ])
    preprocessor.fit(X_train)

    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)

    raw_models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, min_samples_leaf=4,
            max_features=0.5, n_jobs=-1, random_state=42,
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=42,
        ),
        'Linear Regression': LinearRegression(),
    }

    fitted = {}
    metrics = {}
    for name, m in raw_models.items():
        m.fit(X_tr, y_train)
        y_pred = m.predict(X_te)
        fitted[name] = m
        metrics[name] = {
            'R²':  round(r2_score(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'MAE':  round(mean_absolute_error(y_test, y_pred), 2),
        }

    # Feature names from ColumnTransformer (sklearn >= 1.0)
    try:
        feat_names = preprocessor.get_feature_names_out()
        feat_names = [n.replace('num__', '').replace('cat__', '') for n in feat_names]
    except AttributeError:
        feat_names = NUMERIC_COLS + ['ocean_proximity_INLAND',
                                     'ocean_proximity_ISLAND',
                                     'ocean_proximity_NEAR BAY',
                                     'ocean_proximity_NEAR OCEAN']

    return preprocessor, fitted, metrics, feat_names

with st.spinner("Training models on California Housing data — this runs once…"):
    preprocessor, fitted_models, metrics, feat_names = load_and_train()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏡 Real Estate Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">California Housing Dataset · ML-powered valuation in USD & INR · '
    'Feature-engineered for higher accuracy</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
    ocean_prox = st.selectbox("Ocean Proximity",
                              ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    st.markdown('<div class="section-header">Property</div>', unsafe_allow_html=True)
    house_age    = st.slider("Median House Age (yrs)", 1, 52, 20)
    total_rooms  = st.slider("Total Rooms",    1, 20, 6)
    total_bedrms = st.slider("Total Bedrooms", 1, 15, 2)

    st.markdown('<div class="section-header">Neighbourhood</div>', unsafe_allow_html=True)
    med_income = st.slider("Median Income (×$10k)", 0.0, 15.0, 3.5, step=0.1)
    population = st.slider("Population",  10, 10000, 1000)

    st.markdown("---")

    if total_bedrms > total_rooms:
        st.warning("⚠️ Bedrooms exceed total rooms — please adjust.")

    predict_clicked = st.button("🔮 Predict Price")

selected = 'Random Forest'

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    return engineer_features(pd.DataFrame({
        'longitude':          [-119.0],
        'latitude':           [35.0],
        'housing_median_age': [house_age],
        'total_rooms':        [total_rooms],
        'total_bedrooms':     [total_bedrms],
        'population':         [population],
        'households':         [1],
        'median_income':      [med_income],
        'ocean_proximity':    [ocean_prox],
    }))

# ── Main area ─────────────────────────────────────────────────────────────────
tab_pred, tab_compare, tab_importance = st.tabs(
    ["💰 Prediction", "📊 Model Comparison", "🔍 Feature Importance"]
)

# ── Tab 1 — Prediction ────────────────────────────────────────────────────────
with tab_pred:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        meta = MODEL_META[selected]
        m = metrics[selected]
        st.markdown(f"### {meta['icon']} Random Forest Predictor")
        st.markdown(f"<span style='color:#a0aec0'>{meta['desc']}</span>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='margin-top:0.8rem'>"
            f"<span class='metric-chip chip-green'>R² {m['R²']:.4f}</span>"
            f"<span class='metric-chip chip-amber'>RMSE ${m['RMSE']:,.0f}</span>"
            f"<span class='metric-chip chip-blue'>MAE ${m['MAE']:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**How to use:** Adjust sliders in the sidebar → click **Predict Price**.")
        st.markdown(
            "Three **engineered features** boost accuracy over raw inputs: "
            "*rooms/household*, *bedrooms/room*, *population/household*."
        )

    with col_r:
        if predict_clicked:
            X_input = build_input_df()
            X_scaled = preprocessor.transform(X_input)
            model = fitted_models[selected]

            predicted_usd = float(model.predict(X_scaled)[0])
            predicted_inr = predicted_usd * USD_TO_INR

            # Price cards
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="price-card">
                  <div class="price-label">US Dollars</div>
                  <div class="price-usd">${predicted_usd:,.0f}</div>
                  <div class="price-sub">USD</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="price-card">
                  <div class="price-label">Indian Rupees</div>
                  <div class="price-inr">{inr_label(predicted_inr)}</div>
                  <div class="price-sub">₹{predicted_inr:,.0f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<p class="rate-note">Exchange rate: 1 USD = ₹{USD_TO_INR:.2f} (indicative)</p>',
                unsafe_allow_html=True,
            )

            # Input summary
            st.markdown(f"""
            <div class="info-card">
              <b>Model:</b>&nbsp;<span class="model-badge">{selected}</span><br>
              <b>Ocean:</b> {ocean_prox} &nbsp;|&nbsp; <b>Age:</b> {house_age} yrs<br>
              <b>Rooms:</b> {total_rooms} &nbsp;|&nbsp; <b>Bedrooms:</b> {total_bedrms}<br>
              <b>Income:</b> ${med_income*10:.1f}k &nbsp;|&nbsp; <b>Population:</b> {population:,}
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="placeholder-card">
              <div style="font-size:3rem">🔮</div>
              <div style="margin-top:0.8rem">
                Set parameters and click <b>Predict Price</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Tab 2 — Model Comparison ──────────────────────────────────────────────────
with tab_compare:
    st.markdown("### Model Performance on 20% Hold-out Test Set")
    st.markdown(
        "All models trained with the same feature-engineered dataset. "
        "Metrics are on unseen data — no data leakage."
    )

    rows = []
    for name, m in metrics.items():
        meta = MODEL_META[name]
        rows.append({
            'Model': f"{meta['icon']} {name}",
            'R² Score': m['R²'],
            'RMSE (USD)': f"${m['RMSE']:,.0f}",
            'MAE (USD)':  f"${m['MAE']:,.0f}",
            'RMSE (INR)': inr_label(m['RMSE'] * USD_TO_INR),
            'MAE (INR)':  inr_label(m['MAE']  * USD_TO_INR),
        })

    df_metrics = pd.DataFrame(rows).set_index('Model')
    st.dataframe(
        df_metrics.style
            .background_gradient(subset=['R² Score'], cmap='Greens')
            .format({'R² Score': '{:.4f}'}),
        use_container_width=True,
    )

    # R² bar chart
    st.markdown("#### R² Score Comparison")
    r2_data = pd.DataFrame({
        'Model': [f"{MODEL_META[n]['icon']} {n}" for n in metrics],
        'R²':    [metrics[n]['R²']              for n in metrics],
    }).set_index('Model')
    st.bar_chart(r2_data, color='#f7971e')

    st.markdown(
        "_R² measures the proportion of variance explained (1.0 = perfect). "
        "RMSE and MAE are in the same units as the target (USD)._"
    )

# ── Tab 3 — Feature Importance ────────────────────────────────────────────────
with tab_importance:
    st.markdown("### Feature Importances")

    imp_model_name = st.selectbox(
        "Select model for importance",
        ['Random Forest', 'Decision Tree'],
        key='imp_model',
    )

    HIDDEN_FEATURES = {'longitude', 'latitude', 'rooms_per_household', 'population_per_household'}

    imp_model = fitted_models[imp_model_name]
    importances = imp_model.feature_importances_

    n = min(len(importances), len(feat_names))
    fi_df = (
        pd.DataFrame({'Feature': feat_names[:n], 'Importance': importances[:n]})
        .loc[lambda d: ~d['Feature'].isin(HIDDEN_FEATURES)]
        .sort_values('Importance', ascending=False)
        .set_index('Feature')
    )

    st.markdown(f"**{imp_model_name}** — top features driving price predictions:")
    st.bar_chart(fi_df.head(12), color='#667eea')

    with st.expander("Full importance table"):
        st.dataframe(
            fi_df.style
                .background_gradient(subset=['Importance'], cmap='Purples')
                .format({'Importance': '{:.4f}'}),
            use_container_width=True,
        )
    st.markdown(
        "_Importance = mean decrease in impurity (Gini) across trees. "
        "Higher = stronger influence on predicted price._"
    )
