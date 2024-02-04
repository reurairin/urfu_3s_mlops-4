from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from clearml import Task
task = Task.init(project_name='MLOps_final', task_name='ml_approach')

df = pd.read_csv('../data/clean.csv')


vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['Text'])


encoder = LabelEncoder()
y = encoder.fit_transform(df['Sentiment'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_ml = accuracy_score(y_test, y_pred)

precision_ml, recall_ml, f1_score_ml, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted')


print(
    f"ML Approach - Accuracy: {accuracy_ml}, Precision: {precision_ml}, Recall: {recall_ml}, F1-Score: {f1_score_ml}")

logger = task.get_logger()
logger.report_scalar(title='evaluation', series='accuracy',
                     value=accuracy_ml, iteration=1)
logger.report_scalar(title='evaluation', series='precision',
                     value=precision_ml, iteration=1)
logger.report_scalar(title='evaluation', series='recall',
                     value=recall_ml, iteration=1)
logger.report_scalar(title='evaluation', series='f1_',
                     value=f1_score_ml, iteration=1)

task.close()
