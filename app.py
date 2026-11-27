import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Clustering",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #151d35 55%, #211b3a 100%);
        color: #f5f3ff;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(11, 16, 32, 0.85); }
    h1, h2, h3 { font-family: 'Outfit', sans-serif !important; color: #f4eaff !important; }
    p, label, .stMarkdown { color: #d8d9e8; }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(192, 132, 252, 0.25);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(124, 58, 237, 0.23), rgba(30, 41, 75, 0.65));
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #c4b5fd; font-size: 0.82rem; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase;
    }
    .hero-title {
        color: #ffffff; font-family: 'Outfit', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800;
        line-height: 1.1; margin: 8px 0;
    }
    .hero-copy { color: #d5d6e8; font-size: 1rem; max-width: 760px; margin-bottom: 0; }

    div[data-testid="stMetric"] {
        background: rgba(35, 45, 75, 0.78);
        border: 1px solid rgba(167, 139, 250, 0.22);
        padding: 17px 19px; border-radius: 16px;
    }
    div[data-testid="stMetricLabel"] { color: #c4c8dc !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Outfit', sans-serif; }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #8b5cf6, #a855f7);
        color: white; border: 0; border-radius: 11px; font-weight: 700; min-height: 44px;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #7c3aed, #9333ea); color: white; border: 0;
    }
    section[data-testid="stSidebar"] { background: #11182b; border-right: 1px solid rgba(167, 139, 250, 0.18); }
    .result-card {
        background: linear-gradient(120deg, rgba(124, 58, 237, 0.25), rgba(30, 41, 75, 0.8));
        border: 1px solid rgba(196, 181, 253, 0.35);
        border-radius: 16px; padding: 20px; margin-top: 16px;
    }
    .footer { color: #9ca3bd; text-align: center; padding: 22px 0 8px 0; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "iris_kmeans_bundle.pkl"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@st.cache_resource
def load_model_bundle(model_path, modified_time):
    """Load the previously trained K-Means model and fitted scaler."""
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "The model file must contain a dictionary with 'model' and 'scaler' keys. "
            "Use the model-saving code provided with this app."
        )

    if "model" not in bundle or "scaler" not in bundle:
        raise ValueError("The model bundle must contain both 'model' and 'scaler'.")

    return bundle


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Unsupervised machine learning</div>
        <div class="hero-title">Iris Flower Clustering</div>
        <p class="hero-copy">
            Use your previously trained K-Means model to assign a new flower
            to a cluster. No CSV dataset is required to run predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model file not found. Add `iris_kmeans_bundle.pkl` to the "
        "same GitHub repository folder as `app.py`."
    )
    st.info(
        "Create the model bundle using the training-and-saving code provided "
        "with this application, then upload the .pkl file to GitHub."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    kmeans_model = bundle["model"]
    scaler = bundle["scaler"]

    # Optional metadata is saved with the model bundle.
    model_features = bundle.get("features", FEATURES)
    cluster_count = int(getattr(kmeans_model, "n_clusters", 3))
    training_score = bundle.get("silhouette_score", None)
    training_rows = bundle.get("training_rows", None)

    if list(model_features) != FEATURES:
        raise ValueError(
            "The feature order in the saved model does not match this app. "
            "Expected: " + ", ".join(FEATURES)
        )

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🌸 Iris Lab")
    st.caption("Pre-trained K-Means model")
    st.markdown("---")
    st.markdown("### Model information")
    st.write("**Algorithm:** K-Means")
    st.write(f"**Clusters:** {cluster_count}")
    st.write("**Features:** 4 flower measurements")
    st.write("**Scaling:** Saved StandardScaler")
    if training_rows is not None:
        st.write(f"**Training records:** {training_rows}")
    st.markdown("---")
    st.info(
        "This app loads your trained model and scaler. "
        "It does not retrain the model or require iris.csv."
    )


# --------------------------------------------------
# MODEL OVERVIEW
# --------------------------------------------------

st.markdown("## Model overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "K-Means")

with col2:
    st.metric("Clusters", cluster_count)

with col3:
    if training_score is not None:
        st.metric("Training silhouette", f"{float(training_score):.3f}")
    else:
        st.metric("Input features", len(FEATURES))

st.caption(
    "Cluster IDs are arbitrary labels generated during training. "
    "They are not confirmed Iris species names."
)

st.markdown("---")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.markdown("## 🌺 Predict a new flower's cluster")
st.write(
    "Enter the flower's measurements in centimetres. The app uses the "
    "same fitted scaler and trained K-Means model saved from your notebook."
)

with st.form("flower_prediction_form"):
    left, right = st.columns(2)

    with left:
        sepal_length = st.number_input(
            "Sepal length (cm)", min_value=0.1, max_value=10.0,
            value=6.8, step=0.1
        )
        sepal_width = st.number_input(
            "Sepal width (cm)", min_value=0.1, max_value=10.0,
            value=2.7, step=0.1
        )

    with right:
        petal_length = st.number_input(
            "Petal length (cm)", min_value=0.1, max_value=10.0,
            value=1.8, step=0.1
        )
        petal_width = st.number_input(
            "Petal width (cm)", min_value=0.1, max_value=10.0,
            value=0.8, step=0.1
        )

    submitted = st.form_submit_button(
        "🔮 Predict flower cluster",
        width="stretch"
    )

if submitted:
    new_flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES
    )

    try:
        scaled_flower = scaler.transform(new_flower)
        predicted_cluster = int(kmeans_model.predict(scaled_flower)[0])

        st.markdown(
            f"""
            <div class="result-card">
                <div style="color:#c4b5fd;font-weight:700;letter-spacing:1px;">
                    PREDICTION RESULT
                </div>
                <div style="font-size:2rem;font-weight:800;color:white;margin-top:8px;">
                    Cluster {predicted_cluster}
                </div>
                <div style="color:#d5d6e8;">
                    Your trained K-Means model assigned this flower to Cluster {predicted_cluster}.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Submitted measurements")
        st.dataframe(new_flower, width="stretch", hide_index=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🌸 Iris Flower Clustering<br>
        Powered by your trained K-Means model · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
