import json
from pathlib import Path

import joblib
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "toxic_classifier.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


@st.cache_resource
def load_model() -> tuple[object, float, str]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run `python scripts/run_classical.py` before starting the app.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, float(metadata["threshold"]), str(metadata["model"])


st.set_page_config(page_title="Toxic comment classifier", page_icon="🛡️")
st.title("Toxic comment classifier")
st.caption("Binary English-language classifier trained on Civil Comments.")

try:
    classifier, threshold, model_name = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

text = st.text_area("Comment", height=160, placeholder="Enter an English comment")
if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Enter a non-empty comment.")
    else:
        score = float(classifier.predict_proba([text])[0, 1])
        prediction = int(score >= threshold)
        if prediction:
            st.error(f"Toxic — score {score:.3f}")
        else:
            st.success(f"Non-toxic — score {score:.3f}")
        st.caption(f"Model: {model_name}; decision threshold: {threshold:.3f}")
