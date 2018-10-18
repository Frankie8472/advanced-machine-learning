import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import f_regression, SelectPercentile
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Import files
X_train = pd.read_csv('input/X_train.csv', index_col='id')
Y_train = pd.read_csv('input/y_train.csv', index_col='id')
X_test = pd.read_csv('input/X_test.csv', index_col='id')

# Standardize features, takes care of potential outliers
scaler = QuantileTransformer()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# Do a feature selection
selector = SelectPercentile(f_regression, percentile=70)
X_train = pd.DataFrame(selector.fit_transform(X_train, Y_train))
X_test = pd.DataFrame(selector.transform(X_test))

# Split data for validation
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

# Perform training with a SVM, take care of class imbalance
clf = SVC(gamma='scale', kernel='rbf', class_weight='balanced')
clf.fit(X_train, Y_train)

# Get validation metrics
#Y_out = clf.predict(X_val)
#print(balanced_accuracy_score(Y_val, Y_out))

# Get results and print to file
Y_results = clf.predict(X_test)
result = pd.DataFrame(data=Y_results, index=X_test.index.values)
result.to_csv("./output/output_nicu_0.csv", index=True, index_label="id", header=['y'])