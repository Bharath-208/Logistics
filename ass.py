import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
import nltk


# Download NLTK data files (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load data
data = pd.read_excel("sam2.xlsx")

# Initial data checks
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Drop rows with missing values in 'Data' or 'Labels'
data.dropna(subset=['Data', 'Labels'], inplace=True)

# Function to clean text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing
data['cleaned_text'] = data['Data'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(data['cleaned_text'])

# Labels
y = data['Labels']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
nb_classifier = MultinomialNB()
nb_classifier.fit(x_train, y_train)

# Predictions
y_pred = nb_classifier.predict(x_test)

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

data['sentiment'] = data['cleaned_text'].apply(get_sentiment)
print(data['sentiment'].value_counts())

# Evaluation
print(classification_report(y_test, y_pred))



