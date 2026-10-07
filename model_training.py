import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

print("Starting logistic regression...")

#Extract the data to feature vector X and labels y
df = pd.read_csv('features_data.csv')
X = df[['avg_r', 'avg_g', 'avg_b', 'std_r', 'std_g', 'std_b', 'edge_sum']].to_numpy()
y = df['label'].to_numpy()

#split data into 70% training and 15% for test and 15% for validation
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3 ,stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5 ,stratify = y_temp, random_state=0)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#Train the model
clf_1 = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)

#Predict using validation set and calculate error
y_pred = clf_1.predict(X_val_scaled)
accuracy = accuracy_score(y_val,y_pred)
val_error = 1 - accuracy_score(y_val, y_pred)
y_train_pred = clf_1.predict(X_train_scaled)
train_error = 1 - accuracy_score(y_train, y_train_pred)

#Confusion matrix
cm= confusion_matrix(y_val,y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=clf_1.classes_, yticklabels=clf_1.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')


#Print
print(f"Valditaion accuracy: {accuracy}")
print(f"Training error: {train_error}")
print(f"Validation error: {val_error}")
print(cm)
plt.show()

