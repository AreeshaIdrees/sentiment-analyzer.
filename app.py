import streamlit as st
import pandas as pd
from transformers import pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

st.title("Sentiment Analyzer | Areesha Idrees | B04-0923-0058")

@st.cache_resource
def load():
    return pipeline("sentiment-analysis", model=MODEL)

clf = load()
st.caption(f"Model: {MODEL}")

mode = st.radio("Mode", ["Single", "Batch (3 inputs)"], horizontal=True)

if mode == "Single":
    txt = st.text_area("Your text:")
    if st.button("Analyze"):
        if not txt.strip():
            st.warning("Please type some text first.")
        else:
            r = clf(txt, truncation=True)[0]
            st.subheader(r["label"])
            st.write(f"Confidence: {r['score']:.1%}")
else:
    t1 = st.text_area("Input 1:")
    t2 = st.text_area("Input 2:")
    t3 = st.text_area("Input 3:")
    if st.button("Analyze all"):
        texts = [t1, t2, t3]
        valid = [(i + 1, t) for i, t in enumerate(texts) if t.strip()]
        if not valid:
            st.warning("Please fill at least one input.")
        else:
            res = clf([t for _, t in valid], truncation=True)
            st.table(pd.DataFrame({
                "Input #": [i for i, _ in valid],
                "Text": [t[:80] for _, t in valid],
                "Label": [r["label"] for r in res],
                "Confidence": [f"{r['score']:.1%}" for r in res],
            }))