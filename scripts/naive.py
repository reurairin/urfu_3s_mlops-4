from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from clearml import Task
task = Task.init(project_name='MLOps_final', task_name='naive')

df = pd.read_csv('../data/clean.csv')
print(df.head())

positive_words = ['happy', 'great', 'love',
                  'good', 'wonderful', 'joy', 'pleased']
negative_words = ['sad', 'bad', 'hate',
                  'terrible', 'horrible', 'pain', 'angry']


def classify_sentiment(text):
    tokens = str(text).split()

    positive_count = sum(word in positive_words for word in tokens)
    negative_count = sum(word in negative_words for word in tokens)

    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        return 'Neutral'


df['Naive_Prediction'] = df['Normalized_Text'].apply(classify_sentiment)

print(df.sample(5))


vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['Text'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


y_pred_naive = encoder.transform(df['Naive_Prediction'])

_, y_pred_naive_test = train_test_split(
    y_pred_naive, test_size=0.2, random_state=42)

accuracy_naive = accuracy_score(y_test, y_pred_naive_test)
precision_naive, recall_naive, f1_naive, _ = precision_recall_fscore_support(
    y_test, y_pred_naive_test, average='weighted')

print(
    f"Naive Approach - Accuracy: {accuracy_naive}, Precision: {precision_naive}, Recall: {recall_naive}, F1-Score: {f1_naive}")

logger = task.get_logger()
logger.report_scalar(title='evaluation', series='accuracy',
                     value=accuracy_naive, iteration=1)
logger.report_scalar(title='evaluation', series='precision',
                     value=precision_naive, iteration=1)
logger.report_scalar(title='evaluation', series='recall',
                     value=recall_naive, iteration=1)
logger.report_scalar(title='evaluation', series='f1_',
                     value=f1_naive, iteration=1)

task.close()
