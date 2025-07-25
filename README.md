# 🎬 Movie Review Sentiment Analysis

This project uses machine learning and NLP techniques to perform sentiment analysis on movie reviews. It classifies reviews as either **positive** or **negative**, providing valuable insights for filmmakers, studios, and content platforms.

---

## 📌 Project Objectives

- Automatically classify movie reviews based on sentiment (positive or negative).
- Use machine learning models to analyze text data and derive patterns.
- Evaluate and compare the performance of various models to determine the best performer.

---

## 🧠 Technologies Used

- **Python**
- **Scikit-learn**
- **NLTK (Natural Language Toolkit)**
- **XGBoost**
- **Matplotlib & Seaborn** for visualizations

---

## 📂 Dataset

- **Source**: IMDb
- **Size**: 50,000 movie reviews (balanced: 25k positive, 25k negative)
- **Format**: CSV with `review` and `sentiment` columns
- **After cleaning**: 49,582 unique entries

---

## 🧹 Data Preprocessing

1. Remove HTML tags using `BeautifulSoup`
2. Expand contractions (e.g., "can't" → "cannot")
3. Remove emojis and punctuation
4. Tokenize text
5. Remove stopwords and lemmatize words
6. Encode labels: `"positive"` → `1`, `"negative"` → `0`

---

## 📊 Exploratory Data Analysis (EDA)

- Word Clouds for positive & negative reviews
- Distribution plots:
  - Review length (characters & words)
  - Average word length per review
- Most common words:
  - Unigram
  - Bigram
  - Trigram

---

## 🧪 Model Training & Evaluation

| Model                          | Accuracy |
|--------------------------------|----------|
| Logistic Regression            | 89.99%   |
| Multinomial Naive Bayes        | 86.35%   |
| Linear Support Vector Classifier (SVC) | **90.68%** |
| XGBoost                       | 86.42%   |

✅ **Best Performance**: `LinearSVC` with ~90.68% accuracy

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**

---

## 📦 Requirements

```bash
pip install numpy pandas seaborn matplotlib nltk scikit-learn xgboost beautifulsoup4 wordcloud

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
