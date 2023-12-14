import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and Explore Data
data = pd.read_csv('dataset/spam.csv')  # Replace 'your_dataset.csv' with the actual file path
print(data.head())

# Preprocess Data
X = data['v2']
y = data['v1'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Use the Model
new_sms = ["Your new SMS goes here."]
new_sms_vectorized = vectorizer.transform(new_sms)
prediction = model.predict(new_sms_vectorized)

print(f'Prediction: {prediction}')
