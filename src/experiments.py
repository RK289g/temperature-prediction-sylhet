import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os

# load data
data = pd.read_csv('data/air_quality_index_dataset.csv')

print(data.head())

# dropping so2 features 
data = data.drop(columns=['SO2', 'NOX'])
data.head()

# split data
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

print(f'Train set shape: {train_set.shape}', f'Test set shape: {test_set.shape}')

# save train and test set
os.makedirs('data', exist_ok=True)

train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)

# ----------------------------

train_set = pd.read_csv('data/train.csv')

# split features and target
X_train = train_set.drop('Temperature', axis=1)
y_train = train_set['Temperature'].copy()

# validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=42)

print(f'Train set shape: {X_train.shape}', f'Validation set shape: {X_val.shape}')
print(f'Train target shape: {y_train.shape}', f'Validation target shape: {y_val.shape}')

# numerical and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes('object').columns.tolist()

print(f'Numerical columns: {num_cols}', f'Categorical columns: {cat_cols}')

# import SimpleImputer, StandardScaler, OrdinalEncoder

num_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# apply imputer and scaler for numerical columns
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')

# Using OrdinalEncoder with handle_unknown='use_encoded_value' and unknown_value=-1
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])

X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

# Combine X_train and y_train to remove rows with NaN in y_train
train_data = pd.concat([X_train, y_train], axis=1)
train_data = train_data.dropna(subset=[y_train.name])

# Separate X_train and y_train again
X_train = train_data.drop(columns=[y_train.name])
y_train = train_data[y_train.name]

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

# create the model
model = RandomForestRegressor(random_state=42)

# fit the model
model.fit(X_train, y_train)

# predict on validation set
y_pred = model.predict(X_val)

# evaluate the model
rmse = root_mean_squared_error(y_val, y_pred)

print(f'RMSE: {rmse}')

# save the model
import joblib
os.makedirs('models', exist_ok=True)

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')
