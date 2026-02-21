"""
Transboundary Air Pollution Intelligence System - Utility Functions
====================================================================

This module contains reusable functions for data processing, feature engineering,
model training, and evaluation used throughout the project.

Author: TransBoundary-Air-Intelligence-ML-GIS Project
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_standardize_csv(filepath: str, 
                            date_column: str = 'date',
                            country_column: str = 'country') -> pd.DataFrame:
    """
    Load and standardize a CSV file with flexible date and country parsing.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date column
    country_column : str
        Name of the country column
        
    Returns:
    --------
    pd.DataFrame
        Standardized dataframe
    """
    df = pd.read_csv(filepath)
    
    # Standardize date column
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Standardize country names
    if country_column in df.columns:
        df[country_column] = df[country_column].str.strip().str.title()
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                         method: str = 'interpolate',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to use: 'interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop'
    columns : List[str], optional
        Columns to apply missing value handling. If None, applies to all numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if method == 'interpolate':
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        elif method == 'ffill':
            df_clean[col] = df_clean[col].fillna(method='ffill')
        elif method == 'bfill':
            df_clean[col] = df_clean[col].fillna(method='bfill')
        elif method == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif method == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_temporal_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Add temporal features with cyclical encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with date column
    date_column : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added temporal features
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Basic temporal features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['weekofyear'] = df[date_column].dt.isocalendar().week
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Season encoding
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       columns: List[str],
                       lags: List[int] = [1, 7, 14, 30],
                       group_column: str = 'country') -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        Lag periods
    group_column : str
        Column to group by (e.g., country)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(group_column)[col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int] = [7, 14, 30, 90],
                           group_column: str = 'country') -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to create rolling features for
    windows : List[int]
        Window sizes
    group_column : str
        Column to group by
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df.groupby(group_column)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_{window}'] = df.groupby(group_column)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'{col}_rolling_max_{window}'] = df.groupby(group_column)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            df[f'{col}_rolling_min_{window}'] = df.groupby(group_column)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
    
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on Earth.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Coordinates of first point
    lat2, lon2 : float
        Coordinates of second point
        
    Returns:
    --------
    float
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def temporal_train_test_split(df: pd.DataFrame, 
                             date_column: str = 'date',
                             test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally to avoid data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Name of date column
    test_size : float
        Proportion of data for test set
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test dataframes
    """
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    
    return train, test


# ============================================================================
# SPATIAL ANALYSIS
# ============================================================================

def calculate_morans_i(values: np.ndarray, spatial_weights: np.ndarray) -> float:
    """
    Calculate Global Moran's I statistic for spatial autocorrelation.
    
    Parameters:
    -----------
    values : np.ndarray
        Values for each spatial unit
    spatial_weights : np.ndarray
        Spatial weights matrix
        
    Returns:
    --------
    float
        Moran's I statistic
    """
    n = len(values)
    mean_val = np.mean(values)
    
    # Numerator
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += spatial_weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
    
    # Denominator
    denominator = np.sum((values - mean_val) ** 2)
    
    # Sum of weights
    W = np.sum(spatial_weights)
    
    if denominator == 0 or W == 0:
        return 0
    
    morans_i = (n / W) * (numerator / denominator)
    return morans_i


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def categorize_feature(feature_name: str) -> str:
    """
    Categorize feature based on its name.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
        
    Returns:
    --------
    str
        Category name
    """
    if 'neighbor' in feature_name.lower():
        return 'Neighbor'
    elif any(x in feature_name.lower() for x in ['lag', 'rolling']):
        return 'Temporal_Lag'
    elif any(x in feature_name.lower() for x in ['month', 'day', 'year', 'season', 'week']):
        return 'Temporal_Calendar'
    elif any(x in feature_name.lower() for x in ['latitude', 'longitude', 'distance']):
        return 'Spatial'
    elif 'country_' in feature_name.lower():
        return 'Country_Identity'
    elif '*' in feature_name or 'interaction' in feature_name.lower():
        return 'Interaction'
    else:
        return 'Other'


# ============================================================================
# DATA EXPORT
# ============================================================================

def export_for_gis(df: pd.DataFrame, 
                  output_path: str,
                  country_column: str = 'country',
                  lat_column: str = 'latitude',
                  lon_column: str = 'longitude') -> None:
    """
    Export aggregated data in GIS-ready format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_path : str
        Path to save the output
    country_column : str
        Name of country column
    lat_column : str
        Name of latitude column
    lon_column : str
        Name of longitude column
    """
    # Aggregate by country
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    gis_export = df.groupby(country_column).agg({
        **{lat_column: 'first', lon_column: 'first'},
        **{col: 'mean' for col in numeric_cols if col not in [lat_column, lon_column]}
    }).reset_index()
    
    # Save in multiple formats
    gis_export.to_csv(output_path + '.csv', index=False)
    gis_export.to_excel(output_path + '.xlsx', index=False)
    
    print(f"✓ GIS data exported to {output_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive information about a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    name : str
        Name to display
    """
    print(f"\n{'='*80}")
    print(f"{name} Information")
    print(f"{'='*80}")
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.dtypes)
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("None")
    print(f"\nSummary Statistics:")
    print(df.describe())
    print(f"{'='*80}\n")


def create_directory_structure(base_path: str = '.') -> None:
    """
    Create standard directory structure for the project.
    
    Parameters:
    -----------
    base_path : str
        Base path for the project
    """
    import os
    
    directories = [
        'processed_data',
        'eda_outputs',
        'model_outputs',
        'shap_outputs',
        'spatial_outputs',
        'final_report'
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
    
    print("✓ Directory structure created")


if __name__ == "__main__":
    print("TransBoundary Air Pollution Intelligence - Utility Functions")
    print("Available functions:")
    print("  - Data Loading: load_and_standardize_csv, handle_missing_values")
    print("  - Feature Engineering: add_temporal_features, create_lag_features, create_rolling_features")
    print("  - Spatial: haversine_distance, calculate_morans_i")
    print("  - Model Evaluation: evaluate_model, temporal_train_test_split")
    print("  - Analysis: categorize_feature")
    print("  - Export: export_for_gis")
    print("  - Utilities: print_dataframe_info, create_directory_structure")
