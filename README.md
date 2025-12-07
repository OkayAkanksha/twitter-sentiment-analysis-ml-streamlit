# 🐦 Twitter Sentiment Analysis using Machine Learning & Streamlit

![App Screenshot](https://github.com/OkayAkanksha/twitter-sentiment-analysis-ml-streamlit/blob/main/Screenshot%202025-12-07%20153253.png)

This project classifies tweets as **Positive** 😀 or **Negative** 😠 using a Machine Learning model trained on the **Sentiment140** dataset.  
Built with **Python, Scikit-Learn, NLP, TF-IDF, and Streamlit UI**.

> **Note:** To download the dataset from Kaggle, create your own `kaggle.json` API token and place it in `~/.kaggle/` or set it via environment variables.

---

## 🚀 Features
- Text preprocessing with stemming and stopword removal
- TF-IDF vectorization for feature extraction
- Logistic Regression ML Model trained on Sentiment140 dataset
- Interactive Web App built using Streamlit
- Displays sentiment prediction and confidence score

---

## 🧠 Tech Stack
| Component | Technology |
|----------|------------|
| Model Training | Python, Scikit-learn, TF-IDF |
| NLP | NLTK (stopwords, stemming) |
| Web UI | Streamlit |
| Dataset | Sentiment140 (Kaggle) |

---

## 📁 Project Structure
```text
├── app.py                    # Streamlit UI
├── trained_model.sav         # Saved ML model from Colab
├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── requirements.txt          # Dependencies for deployment
└── Twitter_Sentiment_Analysis_using_ML.ipynb  # Training notebook

---

**## ▶️ Run this project locally
**
```bash
git clone https://github.com/OkayAkanksha/twitter-sentiment-analysis-ml-streamlit.git
cd twitter-sentiment-analysis-ml-streamlit
pip install -r requirements.txt
python -m nltk.downloader stopwords
streamlit run app.py

---

**## Dataset Link**
https://www.kaggle.com/datasets/kazanova/sentiment140


