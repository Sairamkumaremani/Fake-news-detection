# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
data = pd.read_csv('D:/Data sets/train.csv')

# Split the data into features and target                   

X = data['text']
y = data['label']
X = X.fillna(" ")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CountVectorizer object
cv = CountVectorizer(stop_words='english')

# Fit and transform the training data using CountVectorizer
X_train_cv = cv.fit_transform(X_train)

# Transform the testing data using CountVectorizer
X_test_cv = cv.transform(X_test)

# Create the decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier
clf.fit(X_train_cv, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test_cv)

# Evaluate the accuracy of the model
accuracy = clf.score(X_test_cv, y_test)
print('Accuracy:', accuracy)

