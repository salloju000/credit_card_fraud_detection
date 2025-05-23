import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Preview the data
print(data.head())
print(data.describe())

# Separate the fraud and valid transactionsA
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

# Calculate fraud ratio
outlierFraction = len(fraud) / float(len(valid))

# Print basic information
print("Fraud Transactions: {}".format(len(fraud)))
print("Valid Transactions: {}".format(len(valid)))

# Amount statistics for both classes
print("Amount details of the fraudulent transactions")
print(fraud.Amount.describe())
print("Amount details of the valid transactions")
print(valid.Amount.describe())

# Correlation heatmap
corrmat = data.corr()
fig = plt.figure(figsize=(14, 10))
sns.heatmap(corrmat, vmax=0.8, square=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()

# Split features and target
x = data.drop(['Class'], axis=1)
y = data['Class']

# Print shape of feature and target data
print(x.shape, y.shape)

# Convert to NumPy arrays
xData = x.values
yData = y.values

# Train-test split: 80% training and 20% testing
xtrain, xtest, ytrain, ytest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Make predictions
ypred = model.predict(xtest)

# Evaluate the model
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)
mcc = matthews_corrcoef(ytest, ypred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Confusion matrix
matrix = confusion_matrix(ytest, ypred)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
