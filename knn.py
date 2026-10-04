import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#read data
df = pd.read_csv('features_data.csv')
X = df[['avg_r', 'avg_g', 'avg_b', 'std_r', 'std_g', 'std_b', 'edge_sum']].to_numpy()
y = df['label'].to_numpy()


#split data as stated
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3 ,stratify = y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.5 ,stratify = y_temp, random_state=0)

#add scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#All k values
k_values = range(1,31)
train_errors = []
val_errors = []


#Iterate through all k values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
    train_errors.append(1 - accuracy_score(y_train, knn.predict(X_train_scaled)))
    val_errors.append(1 - accuracy_score(y_val, knn.predict(X_val_scaled)))

#show best k
best_index = val_errors.index(min(val_errors))
best_k = k_values[best_index]
print("Best k:", best_k)
print("Training error:", train_errors[best_index])
print("Validation error:", val_errors[best_index])


## need to plot


## need to add conf mat