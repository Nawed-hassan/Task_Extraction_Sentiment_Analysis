import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset (Using a sample IMDb dataset)
def load_dataset():
    
    df = pd.read_csv("IMDB Dataset.csv")
    df = df[['review', 'sentiment']]
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df.dropna(subset=['review'], inplace=True)
    return df

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, float):  # Handle NaN values (float)
        return ""  
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters & digits
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)


# Train and evaluate the model
def train_model():
    df = load_dataset()
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)
    
    # Convert text into TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
