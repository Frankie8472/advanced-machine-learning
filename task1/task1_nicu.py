import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import files
X_train_missing = pd.read_csv('input/X_train.csv', index_col='id')
Y_train = pd.read_csv('input/y_train.csv', index_col='id')
X_test_missing = pd.read_csv('input/X_test.csv', index_col='id')

# Impute missing values with median value
imp = SimpleImputer(missing_values=np.nan, strategy='median')
X_train = pd.DataFrame(imp.fit_transform(X_train_missing))
X_train.columns = X_train_missing.columns
X_test = pd.DataFrame(imp.transform(X_test_missing))
X_test.columns = X_test_missing.columns

# Standardize features, takes care of outliers
scaler = QuantileTransformer()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# Do a feature selection
selector = SelectKBest(f_regression, k=222)
X_train = pd.DataFrame(selector.fit_transform(X_train, Y_train))
X_test = pd.DataFrame(selector.transform(X_test))

# Split data for validation
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

# Perform training with an Elastic Net and cross-validation
alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
reg = RidgeCV(alphas=alphas, fit_intercept=True, normalize=False)
reg.fit(X_train, Y_train)

# Get validation metrics
#Y_out = reg.predict(X_val)
#print(r2_score(Y_val, Y_out))

# Get results and print to file
Y_results = reg.predict(X_test)
result = pd.DataFrame(data=Y_results, index=X_test.index.values)
result.to_csv("./output/output_nicu_version_5.csv", index=True, index_label="id", header=['y'])