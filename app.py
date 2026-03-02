"""
app.py — Immo Predictor · Streamlit Application
================================================
Entry point for the Streamlit interface.
All business logic and model loading is delegated to utils.py.

Run
---
    streamlit run src/app.py
"""

from __future__ import annotations
from utils import (
    CLF_FIELD_CONFIG,
    REG_FIELD_CONFIG,
    BLDGTYPE_LABELS,
    NEIGHBORHOOD_LABELS,
    HOUSESTYLE_LABELS,
    confidence_band,
    format_price,
    load_models,
    predict_building_type,
    predict_price,
    validate_year_remod,
)

import sys
from pathlib import Path

import streamlit as st

# Make sure project root is on the path so utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Page configuration
st.set_page_config(
    page_title="Immo Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS — minimal, professional
st.markdown(
    """
   <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap');
    /* Main background */
    *, *::before, *::after {
        font-family: "Lato", "sans-serif";
    }
    
    div[data-testid="stColumn"] {
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }

    .stApp {
        background: linear-gradient(135deg, #3128b4 0%, #3613a8 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.2) 0%, rgba(78, 67, 228, 0.85) 70%, rgba(255, 255, 255, 0.2) 95%);
    }

    /* Headers */
    h1, h2, h3 {
        color: #f8f9fa !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Text */
    p, span, div {
        color: #f8f9fa !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #e94e39;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 36px;
        font-weight: 600;
        transition: ease-in-out 0.5s;
    }

    .stButton>button:hover {
        background-color: #e94e39;
        transform: scale(1.02);
        opacity: 0.95;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* Radios */
    # .stRadio {
    #     margin: .5rem;
    #     font-weight: 900 !important;
    #     font-size: 2.1rem !important;
    # }

    .stb3>label {
        margin-top: .5rem;
    }

    /* Text inputs */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid #756bff;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }

    /* Cards/containers */
    .element-container {
        # background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar — navigation and model info
with st.sidebar:
    st.title("Immo Predictor")
    st.caption("Ames Housing · Machine Learning")
    st.divider()

    task = st.radio(
        "Mode de prédiction",
        options=["Estimation du prix", "Identification du type"],
        help=(
            "**Estimation du prix** — Régression : prédit SalePrice ($).\n\n"
            "**Identification du type** — Classification : prédit BldgType (5 classes)."
        ),
    )

    st.divider()

    # Show model performance cards in sidebar
    try:
        reg_bundle, clf_bundle = load_models()
        if task == "Estimation du prix":
            m = reg_bundle["meta"]
            st.markdown("**Modèle actif**")
            st.caption(m["model_name"])
            col1, col2 = st.columns(2)
            col1.metric("R²",   f"{m['test_r2']:.4f}")
            col2.metric("MAE",  format_price(m["test_mae"]))
            st.metric("RMSE", format_price(m["test_rmse"]))
        else:
            m = clf_bundle["meta"]
            st.markdown("**Modèle actif**")
            st.caption(m["model_name"])
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{m['test_accuracy']:.0%}")
            col2.metric("F1 weighted", f"{m['test_f1_weighted']:.2f}")
    except FileNotFoundError as exc:
        st.error(str(exc))

    st.divider()
    st.caption(
        "Dataset : [Kaggle — House Prices (Ames, Iowa)]"
        "(https://www.kaggle.com/datasets/lespin/house-prices-dataset/data)"
    )


# Helper — build form widgets from field config

def _render_field(col, key: str, cfg: dict) -> object:
    """Render a single input widget and return its value."""
    ftype = cfg["type"]
    label = cfg["label"]
    help_ = cfg.get("help", "")

    if ftype == "slider":
        return col.slider(
            label,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            help=help_,
        )
    elif ftype == "number":
        return col.number_input(
            label,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg.get("step", 1),
            help=help_,
        )
    elif ftype == "select":
        options = cfg["options"]
        labels_map = cfg.get("labels", {})
        display_labels = [labels_map.get(o, o) for o in options]
        default_idx = options.index(
            cfg["default"]) if cfg["default"] in options else 0
        selected_label = col.selectbox(
            label,
            options=display_labels,
            index=default_idx,
            help=help_,
        )
        # Reverse-map display label to code
        reverse = {v: k for k, v in labels_map.items()}
        return reverse.get(selected_label, selected_label)
    else:
        return col.text_input(label, value=str(cfg["default"]), help=help_)


# Main content
if task == "Estimation du prix":
    st.header("Estimation du Prix de Vente")
    st.markdown(
        "Renseignez les caractéristiques du bien pour obtenir une estimation du prix de vente "
        "à Ames, Iowa. Les champs marqués d'un `*` ont le plus fort impact sur le prix."
    )
    st.divider()

    with st.form("form_regression"):
        st.subheader("Caractéristiques du bien")

        # Row 1 — Quality & size (most impactful)
        c1, c2 = st.columns(2)
        inputs = {}
        inputs["OverallQual"] = _render_field(
            c1, "OverallQual",   REG_FIELD_CONFIG["OverallQual"])
        inputs["GrLivArea"] = _render_field(
            c2, "GrLivArea",     REG_FIELD_CONFIG["GrLivArea"])

        c1, c2 = st.columns(2)
        inputs["TotalBsmtSF"] = _render_field(
            c1, "TotalBsmtSF",   REG_FIELD_CONFIG["TotalBsmtSF"])
        inputs["GarageArea"] = _render_field(
            c2, "GarageArea",    REG_FIELD_CONFIG["GarageArea"])

        c1, c2 = st.columns(2)
        inputs["GarageCars"] = _render_field(
            c1, "GarageCars",    REG_FIELD_CONFIG["GarageCars"])
        inputs["LotArea"] = _render_field(
            c2, "LotArea",       REG_FIELD_CONFIG["LotArea"])

        st.markdown("**Localisation & Époque**")
        c1, c2 = st.columns(2)
        inputs["Neighborhood"] = _render_field(
            c1, "Neighborhood",  REG_FIELD_CONFIG["Neighborhood"])
        inputs["YearBuilt"] = _render_field(
            c2, "YearBuilt",     REG_FIELD_CONFIG["YearBuilt"])

        c1, c2 = st.columns(2)
        inputs["YearRemodAdd"] = _render_field(
            c1, "YearRemodAdd",  REG_FIELD_CONFIG["YearRemodAdd"])
        inputs["OverallCond"] = _render_field(
            c2, "OverallCond",   REG_FIELD_CONFIG["OverallCond"])

        st.markdown("**Pièces & Équipements**")
        c1, c2, c3 = st.columns(3)
        inputs["BedroomAbvGr"] = _render_field(
            c1, "BedroomAbvGr",  REG_FIELD_CONFIG["BedroomAbvGr"])
        inputs["FullBath"] = _render_field(
            c2, "FullBath",      REG_FIELD_CONFIG["FullBath"])
        inputs["TotRmsAbvGrd"] = _render_field(
            c3, "TotRmsAbvGrd",  REG_FIELD_CONFIG["TotRmsAbvGrd"])

        c1, c2 = st.columns(2)
        inputs["Fireplaces"] = _render_field(
            c1, "Fireplaces",    REG_FIELD_CONFIG["Fireplaces"])
        inputs["PoolArea"] = _render_field(
            c2, "PoolArea",      REG_FIELD_CONFIG["PoolArea"])

        submitted = st.form_submit_button(
            "Estimer le prix", type="primary", use_container_width=True)

    if submitted:
        # Validate year coherence
        inputs["YearRemodAdd"] = validate_year_remod(
            int(inputs["YearBuilt"]), int(inputs["YearRemodAdd"])
        )

        with st.spinner("Calcul en cours…"):
            try:
                price, meta = predict_price(inputs)
                low, high = confidence_band(meta["test_mae"], price)

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h2>Prix estimé : {format_price(price)}</h2>
                        <p>Intervalle de confiance (±MAE) : {format_price(low)} — {format_price(high)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("R² (test)",  f"{meta['test_r2']:.4f}")
                col2.metric("MAE",        format_price(meta["test_mae"]))
                col3.metric("RMSE",       format_price(meta["test_rmse"]))

                st.caption(
                    f"Modèle : {meta['model_name']} · "
                    "Dataset : Ames Housing (Kaggle, 2006–2010)"
                )
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")


# Classification form
else:
    st.header("Identification du Type de Bâtiment")
    st.markdown(
        "Renseignez les caractéristiques du bien. Le modèle prédit le type de bâtiment "
        "(*BldgType*) parmi 5 catégories, avec probabilités par classe."
    )
    st.divider()

    with st.form("form_classification"):
        st.subheader("Caractéristiques du bien")

        c1, c2 = st.columns(2)
        inputs = {}
        inputs["OverallQual"] = _render_field(
            c1, "OverallQual",  CLF_FIELD_CONFIG["OverallQual"])
        inputs["GrLivArea"] = _render_field(
            c2, "GrLivArea",    CLF_FIELD_CONFIG["GrLivArea"])

        c1, c2 = st.columns(2)
        inputs["TotRmsAbvGrd"] = _render_field(
            c1, "TotRmsAbvGrd", CLF_FIELD_CONFIG["TotRmsAbvGrd"])
        inputs["GarageCars"] = _render_field(
            c2, "GarageCars",   CLF_FIELD_CONFIG["GarageCars"])

        c1, c2 = st.columns(2)
        inputs["YearBuilt"] = _render_field(
            c1, "YearBuilt",    CLF_FIELD_CONFIG["YearBuilt"])

        st.markdown("**Localisation & Style**")
        c1, c2 = st.columns(2)
        inputs["Neighborhood"] = _render_field(
            c1, "Neighborhood", CLF_FIELD_CONFIG["Neighborhood"])
        inputs["HouseStyle"] = _render_field(
            c2, "HouseStyle",   CLF_FIELD_CONFIG["HouseStyle"])

        submitted = st.form_submit_button(
            "Identifier le type", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Classification en cours…"):
            try:
                pred_class, probs, classes, meta = predict_building_type(
                    inputs)
                pred_label = BLDGTYPE_LABELS.get(pred_class, pred_class)
                max_prob = probs[classes.index(pred_class)]

                st.markdown(
                    f"""
                    <div class="result-box">
                        <h2>Type prédit : {pred_label}</h2>
                        <p>Confiance : {max_prob:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("Probabilités par classe")
                prob_df = sorted(zip(classes, probs),
                                 key=lambda x: x[1], reverse=True)
                for cls, prob in prob_df:
                    label = BLDGTYPE_LABELS.get(cls, cls)
                    col1, col2 = st.columns([3, 7])
                    col1.markdown(
                        f"<span class='prob-label'>{label}</span>", unsafe_allow_html=True)
                    col2.progress(float(prob), text=f"{prob:.1%}")

                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (test)",
                            f"{meta['test_accuracy']:.0%}")
                col2.metric("F1 weighted (test)",
                            f"{meta['test_f1_weighted']:.2f}")

                st.caption(
                    f"Modèle : {meta['model_name']} · "
                    "Dataset : Ames Housing (Kaggle, 2006–2010)"
                )

                # Note on minority classes
                if pred_class in ("2fmCon", "Duplx", "Twnhs"):
                    st.info(
                        f"**Note :** La classe `{pred_class}` est rare dans le dataset d'entraînement "
                        f"(<4% des observations). La fiabilité de la prédiction est réduite pour "
                        "ces catégories."
                    )

            except Exception as exc:
                st.error(f"Erreur lors de la classification : {exc}")
