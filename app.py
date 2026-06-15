import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 AI Text Summarizer")
st.write("Paste a long piece of text below and get a short, AI-generated summary.")
st.caption("Powered by a T5 model fine-tuned on the CNN/DailyMail dataset.")

MODEL_NAME = "levi1234/t5-small-summarizer-cnn"


# Load the summarization model (this downloads once and is cached afterwards)
@st.cache_resource
def load_model():
    return pipeline("summarization", model=MODEL_NAME)


with st.spinner("Loading AI model... (this may take a minute the first time)"):
    summarizer = load_model()

# Text input
text = st.text_area(
    "Paste your text here:",
    height=300,
    placeholder="Paste an article, essay, or long paragraph here...",
)

# Summary length controls
col1, col2 = st.columns(2)
with col1:
    min_len = st.slider("Minimum summary length (words)", 10, 100, 30)
with col2:
    max_len = st.slider("Maximum summary length (words)", 50, 300, 130)

# Summarize button
if st.button("Summarize"):
    if not text.strip():
        st.warning("Please paste some text first.")
    elif len(text.split()) < 20:
        st.warning(
            "Please paste a longer piece of text (at least ~20 words) for a meaningful summary."
        )
    else:
        with st.spinner("Summarizing..."):
            # T5 models expect a "summarize: " prefix
            result = summarizer(
                "summarize: " + text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            summary = result[0]["summary_text"]

        st.subheader("Summary")
        st.success(summary)

        # Show some stats
        st.caption(
            f"Original length: {len(text.split())} words → Summary length: {len(summary.split())} words"
        )
