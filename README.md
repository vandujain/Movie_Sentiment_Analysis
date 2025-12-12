🎬 Movie Sentiment Analysis

Sentiment Classification on IMDB Movie Reviews using NLP + Machine Learning








📌 Overview

This project performs sentiment analysis on 50,000 IMDB movie reviews using traditional NLP techniques and machine learning models.
I implemented text preprocessing, TF-IDF vectorization, and multiple classification models, comparing performance across algorithms.

📂 Dataset

Source: IMDB Movie Review Dataset

Total Reviews: 50,000

Sentiments: Balanced (25k positive, 25k negative)

Columns:

review (text)

sentiment (positive/negative → encoded as 1/0)

🧠 Project Workflow
1. Loading Dataset  
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. Feature Engineering (TF-IDF)
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Results & Comparison

🔍 Exploratory Data Analysis

Sentiment distribution

Review length statistics

Word count histograms

Boxplots by sentiment

Longest & shortest reviews

WordCloud of raw text

Binning review lengths (Short → Extreme)

🧹 Text Preprocessing Pipeline

✔ Expand contractions
✔ Lowercasing
✔ Remove HTML tags
✔ Remove URLs
✔ Remove emojis
✔ Remove punctuation
✔ Tokenization
✔ Remove stopwords (keep negations)
✔ POS-based lemmatization
✔ Join tokens back
✔ Encode labels

Sample Preprocessing Code
contractions.fix("can't")  # cannot

html_tags = re.compile(r'<.*?>')
clean_text = html_tags.sub(" ", text)

tokens = word_tokenize(clean_text.lower())
filtered_tokens = [w for w in tokens if w not in stop_words]

lemmatizer.lemmatize(w, pos='v')

🧪 Feature Extraction — TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data['lemmatized'])

🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

🤖 Models Trained
Model	Description
Logistic Regression	Baseline linear classifier
Naive Bayes	Probabilistic text model
Linear SVM	High-margin classifier
XGBoost	Gradient boosting ensemble
📊 Model Performance
Accuracy Scores
Model	Accuracy
Logistic Regression	⭐ 0.8946
Linear SVM	⭐ 0.8945
XGBoost	0.8635
Naive Bayes	0.8617
🔥 Best Models

Logistic Regression

Linear SVM

Both achieve ~89.4% accuracy.

📈 Visualizations

Included in the notebook:

📊 Sentiment count plot

🔠 Word/character count distribution

📦 Boxplots

☁️ WordClouds

🔡 Top frequent words (before/after cleaning)

🧮 Confusion matrices

📉 Accuracy comparison barplot

🧾 Conclusions

Traditional ML + TF-IDF is highly effective for sentiment tasks.

Proper text preprocessing massively improves model quality.

Logistic Regression & Linear SVM perform best, matching literature benchmarks.

🚀 Future Improvements

Fine-tuning BERT / DistilBERT

LSTM/GRU-based models

Hyperparameter tuning

Deploying model using Streamlit / FastAPI

LIME/SHAP interpretability

📁 Project Structure
📦 Movie-Sentiment-Analysis
├── 📄 IMDB Dataset.csv
├── 📓 MOVIE_SENTIMENT_ANALYSIS.ipynb
├── 📝 README.md
└── 📊 results/
     ├── confusion_matrices/
     ├── wordcloud.png
     └── accuracy_plot.png

✨ Author

Vandana Jain
MSc Data Science | NLP & Machine Learning Enthusiast
