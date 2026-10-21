import os
import pickle
import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TruthScan AI | Next-Gen Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "train.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset_predictions.csv")

# ---------------- ULTRA-MODERN CSS DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080d1a 45%, #030712 100%);
    color: #e2e8f0;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(11, 18, 33, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
}

/* Glassmorphism Hero */
.hero-container {
    padding: 34px 38px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.45) 0%, rgba(15, 23, 42, 0.6) 50%, rgba(13, 148, 136, 0.25) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}

.hero-title {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-badge {
    display: inline-block;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 8px;
    max-width: 720px;
    line-height: 1.5;
}

/* Metric Cards with Hover Glow */
.metric-card-modern {
    padding: 22px;
    border-radius: 20px;
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    transition: all 0.25s ease-in-out;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.metric-card-modern:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-icon-badge {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    background: rgba(255, 255, 255, 0.05);
}

.metric-val {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 800;
    margin-top: 10px;
    letter-spacing: -0.02em;
}

.metric-sub {
    color: #64748b;
    font-size: 12px;
    margin-top: 4px;
}

/* Prediction Result Banners */
.result-card {
    padding: 28px;
    border-radius: 22px;
    backdrop-filter: blur(14px);
    margin-top: 20px;
    animation: fadeIn 0.4s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-card.real {
    background: radial-gradient(circle at top left, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
    border: 1px solid rgba(52, 211, 153, 0.45);
    box-shadow: 0 15px 30px -10px rgba(16, 185, 129, 0.2);
}

.result-card.fake {
    background: radial-gradient(circle at top left, rgba(239, 68, 68, 0.15) 0%, rgba(136, 19, 55, 0.25) 100%);
    border: 1px solid rgba(251, 113, 133, 0.45);
    box-shadow: 0 15px 30px -10px rgba(239, 68, 68, 0.2);
}

.result-tag {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}

.tag-real {
    background: rgba(52, 211, 153, 0.2);
    color: #4ade80;
    border: 1px solid rgba(52, 211, 153, 0.4);
}

.tag-fake {
    background: rgba(251, 113, 133, 0.2);
    color: #fb7185;
    border: 1px solid rgba(251, 113, 133, 0.4);
}

/* Custom inputs & buttons */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: rgba(17, 24, 39, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CACHED MODEL LOADING ----------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


def combine_text(df):
    title = df["title"].fillna("").astype(str) if "title" in df.columns else pd.Series("", index=df.index)
    article = df["text"].fillna("").astype(str) if "text" in df.columns else pd.Series("", index=df.index)
    return (title + " " + article).str.strip()


def label_to_text(value):
    val_str = str(value).strip().lower()
    if val_str in ("0", "real", "true", "legitimate"):
        return "REAL"
    if val_str in ("1", "fake", "false"):
        return "FAKE"
    return val_str.upper()


def make_predictions(data, model, vectorizer):
    text_data = combine_text(data)
    transformed = vectorizer.transform(text_data)
    predictions = model.predict(transformed)

    result = data.copy()
    result["model_prediction"] = predictions
    result["predicted_news_type"] = [label_to_text(v) for v in predictions]

    if "label" in result.columns:
        result["actual_news_type"] = result["label"].apply(label_to_text)
        result["prediction_matches_label"] = (
            result["actual_news_type"] == result["predicted_news_type"]
        )
    return result


def metric_box(label, value, note, icon, border_accent=None):
    border_style = f"border-top: 3px solid {border_accent};" if border_accent else ""
    st.markdown(f"""
    <div class="metric-card-modern" style="{border_style}">
        <div class="metric-header">
            <span>{label}</span>
            <div class="metric-icon-badge">{icon}</div>
        </div>
        <div class="metric-val">{value}</div>
        <div class="metric-sub">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 26px;">🛡️</span>
        <span style="font-size: 20px; font-weight: 800; color: #ffffff;">TruthScan AI</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("AI-Powered Information Integrity Engine")
    st.write("")

    page = st.radio(
        "NAVIGATION",
        [
            "📊 Dashboard",
            "🔍 Check News",
            "📁 Dataset Explorer",
            "ℹ️ About Engine"
        ]
    )

    st.write("---")
    st.markdown("<p style='font-size: 12px; font-weight: 700; color: #94a3b8; margin-bottom: 8px;'>SYSTEM INTEGRITY</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            st.success("Artifacts OK")
        else:
            st.error("Artifacts Missing")
    with c2:
        if os.path.exists(DATASET_PATH):
            st.success("Data Linked")
        else:
            st.warning("No Data")

    st.write("---")
    st.caption("Model: TF-IDF + Classifier · Ver. 2.1")


# ---------------- LOAD ARTIFACTS ----------------
try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Model load nahi ho paya. `artifacts/` folder aur pickle files check karo.")
    st.code(str(e))
    st.stop()


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Deep Text Semantics</div>
    <div class="hero-title">TruthScan AI Platform</div>
    <div class="hero-subtitle">
        Automated classification and veracity screening for news articles using natural language processing and statistical linguistic modeling.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- PROCESS DATASET ----------------
if os.path.exists(DATASET_PATH):
    try:
        data = load_dataset()
        if os.path.exists(OUTPUT_PATH):
            results = pd.read_csv(OUTPUT_PATH)
            if not {"predicted_news_type", "model_prediction"}.issubset(results.columns):
                results = make_predictions(data, model, vectorizer)
                results.to_csv(OUTPUT_PATH, index=False)
        else:
            with st.spinner("Generating initial batch predictions..."):
                results = make_predictions(data, model, vectorizer)
                results.to_csv(OUTPUT_PATH, index=False)
    except Exception as e:
        st.error("Dataset load error:")
        st.code(str(e))
        results = pd.DataFrame()
else:
    data = pd.DataFrame()
    results = pd.DataFrame()


# ---------------- PAGE: DASHBOARD ----------------
if page == "📊 Dashboard":
    if results.empty:
        st.warning("train.csv file nahi mili. Project root folder mein CSV add karein.")
    else:
        total = len(results)
        real_count = int((results["predicted_news_type"] == "REAL").sum())
        fake_count = int((results["predicted_news_type"] == "FAKE").sum())
        real_pct = (real_count / total * 100) if total else 0
        fake_pct = (fake_count / total * 100) if total else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_box("Total Corpus", f"{total:,}", "Articles evaluated in dataset", "📰", "#3b82f6")
        with c2:
            metric_box("Verified Real", f"{real_count:,}", f"{real_pct:.1f}% share of dataset", "🟢", "#10b981")
        with c3:
            metric_box("Flagged Fake", f"{fake_count:,}", f"{fake_pct:.1f}% share of dataset", "🔴", "#f43f5e")

        st.write("")
        st.write("")

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("#### 📊 Prediction Volume")
            chart_df = pd.DataFrame({
                "Count": [real_count, fake_count]
            }, index=["REAL", "FAKE"])
            st.bar_chart(chart_df, color="#6366f1")

        with col_right:
            st.markdown("#### 🎯 Distribution Share")
            pct_df = pd.DataFrame({
                "Share (%)": [real_pct, fake_pct]
            }, index=["REAL", "FAKE"])
            st.bar_chart(pct_df, color="#10b981")

        st.write("---")
        st.markdown("#### 📑 Sample Corpus Inferences")

        display_cols = [c for c in ["title", "author", "predicted_news_type", "actual_news_type", "prediction_matches_label"] if c in results.columns]
        st.dataframe(results[display_cols].head(8), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Export Inferred Corpus (.csv)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="truthscan_dataset_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------- PAGE: CHECK NEWS ----------------
elif page == "🔍 Check News":
    st.markdown("### 🔍 Real-Time Article Inspection")
    st.caption("Paste article text below to trigger classification.")

    col1, col2 = st.columns([3, 1])
    with col1:
        title_in = st.text_input("Article Headline", placeholder="e.g., Global Summit Reaches Landmark Accord on Climate Targets")
    with col2:
        source_in = st.text_input("Source / Outlet (Optional)", placeholder="e.g., Reuters, BBC")

    article_in = st.text_area("Full Article Text", placeholder="Paste body paragraph here...", height=220)

    if st.button("⚡ Run Veracity Scan", use_container_width=True):
        full_text = f"{title_in} {article_in}".strip()

        if not full_text:
            st.warning("Kripya kam se kam headline ya article body enter karein.")
        else:
            with st.spinner("Analyzing linguistic structures & vector patterns..."):
                transformed = vectorizer.transform([full_text])
                prediction = model.predict(transformed)[0]
                news_type = label_to_text(prediction)

                # Probabilities if available
                has_prob = hasattr(model, "predict_proba")
                real_score = None
                fake_score = None

                if has_prob:
                    probs = model.predict_proba(transformed)[0]
                    classes = [label_to_text(c) for c in model.classes_]
                    for cls_label, p in zip(classes, probs):
                        if cls_label == "REAL":
                            real_score = p * 100
                        elif cls_label == "FAKE":
                            fake_score = p * 100

                if news_type == "REAL":
                    st.markdown(f"""
                    <div class="result-card real">
                        <span class="result-tag tag-real">VERIFIED CHARACTERISTICS</span>
                        <h2 style="color: #34d399; margin: 0; font-size: 26px; font-weight: 800;">VERDICT: REAL NEWS</h2>
                        <p style="color: #a7f3d0; margin-top: 6px; font-size: 14px;">
                            The linguistic features and vocabulary patterns align with standard verified reporting.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card fake">
                        <span class="result-tag tag-fake">DECEPTIVE SIGNALS DETECTED</span>
                        <h2 style="color: #fb7185; margin: 0; font-size: 26px; font-weight: 800;">VERDICT: SUSPECTED FAKE NEWS</h2>
                        <p style="color: #fecdd3; margin-top: 6px; font-size: 14px;">
                            High concentration of sensational or uncharacteristic phrasing detected by the inference engine.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                if has_prob and real_score is not None and fake_score is not None:
                    st.write("")
                    st.markdown("##### 🎯 Model Confidence Spread")
                    c_real, c_fake = st.columns(2)
                    with c_real:
                        st.write(f"**Real Confidence:** `{real_score:.1f}%`")
                        st.progress(float(real_score) / 100.0)
                    with c_fake:
                        st.write(f"**Fake Confidence:** `{fake_score:.1f}%`")
                        st.progress(float(fake_score) / 100.0)

                st.caption("Disclaimer: AI models detect syntactic and lexical correlations, not ground truth facts. Critical claims should be cross-verified.")


# ---------------- PAGE: DATASET EXPLORER ----------------
elif page == "📁 Dataset Explorer":
    st.markdown("### 📁 Dataset Inspection & Search")
    st.caption("Deep-dive into train dataset predictions with lexical filtering.")

    if results.empty:
        st.warning("No records available to display.")
    else:
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            filter_type = st.selectbox("Filter Class", ["ALL", "REAL", "FAKE"])
        with fc2:
            search_text = st.text_input("Search In Headlines/Content", placeholder="Search keyword...")

        filtered = results.copy()
        if filter_type != "ALL":
            filtered = filtered[filtered["predicted_news_type"] == filter_type]

        if search_text:
            text_cols = [c for c in ["title", "text", "author"] if c in filtered.columns]
            if text_cols:
                mask = filtered[text_cols].fillna("").astype(str).apply(
                    lambda col: col.str.contains(search_text, case=False, regex=False)
                ).any(axis=1)
                filtered = filtered[mask]

        st.caption(f"Showing **{len(filtered):,}** matching entries")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Filtered Results",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------- PAGE: ABOUT ----------------
elif page == "ℹ️ About Engine":
    st.markdown("### ℹ️ Architecture & Pipeline")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="metric-card-modern" style="min-height: 220px;">
            <div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">⚙️ Pipeline Architecture</div>
            <p style="color: #94a3b8; font-size: 13px; line-height: 1.6;">
                1. <b>Text Sanitization:</b> Title and body merging with missing value imputation.<br>
                2. <b>TF-IDF Vectorization:</b> Converts raw text into n-gram statistical frequency matrices.<br>
                3. <b>Classifier Inference:</b> Linear or ensemble decision boundaries determine class assignment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div class="metric-card-modern" style="min-height: 220px;">
            <div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">🛠️ Tech Stack</div>
            <p style="color: #94a3b8; font-size: 13px; line-height: 1.6;">
                • <b>Engine:</b> Python & Scikit-learn<br>
                • <b>Interface:</b> Streamlit Custom CSS Framework<br>
                • <b>Data Processing:</b> Pandas Vectorized Ops<br>
                • <b>Deployment:</b> Streamlit Cloud / HuggingFace Spaces compatible
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 12px;'>"
    "TruthScan AI · Built for news veracity screening · All outputs require contextual review."
    "</p>",
    unsafe_allow_html=True
)