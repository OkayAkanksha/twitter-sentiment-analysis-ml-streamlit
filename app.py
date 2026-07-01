import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (only first run will actually download)
nltk.download('stopwords', quiet=True)

# === Load the model and vectorizer ===
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_artifacts()

# === Preprocessing (same as your notebook) ===
pattern = re.compile('[^a-zA-Z]')
english_stopwords = stopwords.words('english')
port_stemmer = PorterStemmer()
TEXT_COLUMN_CANDIDATES = {
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "body",
}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 5000

def stemming(content: str) -> str:
    stemmed_content = re.sub(pattern, ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stemmer.stem(word)
        for word in stemmed_content
        if word not in english_stopwords
    ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def find_text_column(columns):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def sentiment_label(prediction):
    if prediction == 1:
        return "Positive"
    return "Negative"


# === Streamlit UI ===
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

st.title("🐦 Twitter Sentiment Analysis App")
st.write("Analyze whether a tweet expresses a positive or negative sentiment using your trained machine learning model.")

tweet_input = st.text_area(
    "Enter a tweet:",
    placeholder="Type or paste any tweet here...",
    height=150
)

if st.button("Analyze Sentiment"):
    if not tweet_input.strip():
        st.warning("Please enter a tweet before clicking the button.")
    else:
        # 1. Preprocess text
        processed_text = stemming(tweet_input)

        # 2. Vectorize
        vectorized_text = vectorizer.transform([processed_text])

        # 3. Predict
        prediction = model.predict(vectorized_text)[0]

        # 4. Confidence (if available)
        try:
            proba = model.predict_proba(vectorized_text)[0]
            confidence = float(np.max(proba))
        except Exception:
            confidence = None

        # 5. Map to label
        if prediction == 1:
            sentiment = "Positive 😀"
        else:
            sentiment = "Negative 😠"

        st.subheader("Result")
        st.markdown(f"**Sentiment:** {sentiment}")
        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence:.2f}")

        with st.expander("Processed tweet (debug info)"):
            st.write(processed_text)

st.markdown("---")
st.subheader("Batch CSV Prediction")
uploaded_file = st.file_uploader("Upload a CSV with a tweet text column", type=["csv"])

if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("CSV is too large. Upload a file up to 2 MB.")
    else:
        try:
            csv_data = pd.read_csv(uploaded_file, nrows=MAX_BATCH_ROWS + 1)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
        else:
            if len(csv_data) > MAX_BATCH_ROWS:
                st.error(f"CSV has more than {MAX_BATCH_ROWS} rows. Trim it and upload again.")
            else:
                text_column = find_text_column(csv_data.columns)
                if text_column is None:
                    st.error("Add a text, tweet, tweet_text, full_text, content, or body column.")
                else:
                    processed = csv_data[text_column].fillna("").astype(str).map(stemming)
                    predictions = model.predict(vectorizer.transform(processed))
                    results = csv_data.copy()
                    results["predicted_sentiment"] = [
                        sentiment_label(prediction) for prediction in predictions
                    ]
                    st.dataframe(results.head(20))
                    st.download_button(
                        "Download predictions",
                        results.to_csv(index=False).encode("utf-8"),
                        "sentiment_predictions.csv",
                        "text/csv",
                    )

st.markdown("---")
st.caption("Model trained on Sentiment140 dataset (0 = negative, 1 = positive).")
