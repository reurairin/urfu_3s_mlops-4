import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# missing values
df = pd.read_csv('../data/train.csv')

df['Text'] = df['Text'].astype(str)

print(df.isnull().sum())

df = df.dropna()

df = df.reset_index(drop=True)


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    return text


df['Text'] = df['Text'].apply(clean_text)

df['Text'] = df['Text'].str.lower()

df = df.dropna(subset=['Sentiment'])

df = df.drop(columns=['Unnamed: 0'])

df = df.reset_index(drop=True)


# normalize text
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


def normalize_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    normalized_tokens = [lemmatizer.lemmatize(
        word) for word in tokens if word not in stop_words]

    normalized_text = ' '.join(normalized_tokens)

    return normalized_text


df['Normalized_Text'] = df['Text'].apply(normalize_text)

df.to_csv('../data/clean.csv', index=False)
