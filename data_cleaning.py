import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def clean_data(df, target):
    y = df[target]
    df = df.drop([target], axis = 1)

    # 1. Drop Columns with >30% Missing Data
    missing_threshold = 0.3
    df = df.loc[:, df.isnull().mean() < missing_threshold]

    # 2. Remove Duplicate Rows
    df = df.drop_duplicates()

    # 3. Handle Null Values
    # - For numeric columns, fill missing values with the median
    # - For categorical columns, fill missing values with the mode
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    cat_columns_to_encode = [col for col in categorical_cols if df[col].nunique() < 10]
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # 4. Encode Categorical Columns with <20 Unique Values
    # Identify columns with fewer than 20 unique values for encoding

    remove_cat_cols = [col for col in categorical_cols if df[col].nunique() >= 10]
    df = df.drop(remove_cat_cols, axis=1)
    # Apply Label Encoding
    label_encoders = {}
    for col in cat_columns_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder if inverse transform is needed

    # 5. Standardize Numeric Features (Optional)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 6. Drop Columns with Zero Variance
    df = df.loc[:, (df != df.iloc[0]).any()]  # Drop zero-variance columns

    if y.dtype == 'object' or y.dtype.name == 'category':
        # Encode categorical target
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        y = pd.DataFrame(y, columns=[target])
    return df, y

