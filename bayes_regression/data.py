import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def load_and_prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    1. Load California Housing dataset
    2. Select a small set of predictors
    3. Train/test split
    4. Standardize predictors (fit scaler on train, transform test)

    This follows the exact steps from the original project.
    """
    # Load California Housing dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    # Select predictors + response (same as in the PDF)
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
    X_raw = df[feature_names].values
    y_raw = df["MedHouseVal"].values

    # Train/test split (same settings)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state
    )

    # Standardize predictors (important for MCMC)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    # Design matrices with intercept column
    X_train = np.column_stack([np.ones(X_train_std.shape[0]), X_train_std])
    X_test = np.column_stack([np.ones(X_test_std.shape[0]), X_test_std])

    # param_names in the notebook: Intercept + original features
    param_names = ["Intercept"] + feature_names

    return X_train, X_test, y_train, y_test, param_names


def fit_ols_baseline(X_train, y_train):
    """
    OLS regression on the standardized design matrix.
    This matches the baseline OLS fit in the original project.
    """
    ols_model = sm.OLS(y_train, X_train).fit()
    return ols_model
