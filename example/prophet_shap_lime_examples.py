"""
Prophet Model Interpretability with SHAP and LIME
==================================================

This script demonstrates how to use SHAP and LIME for interpreting Prophet model predictions.
We'll create a Prophet model for time series forecasting and apply both explainability methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import shap
import lime
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# SECTION 1: Data Generation and Preprocessing
# ============================================================================

def generate_synthetic_timeseries_for_prophet(n_days=1000):
    """
    Generate synthetic time series data suitable for Prophet modeling.
    
    Args:
        n_days: Number of days to generate
    
    Returns:
        DataFrame with 'ds' (date) and 'y' (value) columns for Prophet
    """
    print("Generating synthetic time series data for Prophet...")
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate synthetic time series with trend, seasonality, and noise
    t = np.arange(n_days)
    
    # Components
    trend = 0.01 * t + 10  # Linear trend
    yearly_seasonality = 3 * np.sin(2 * np.pi * t / 365.25)  # Yearly cycle
    weekly_seasonality = 1.5 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
    monthly_seasonality = 0.8 * np.sin(2 * np.pi * t / 30.44)  # Monthly cycle
    noise = np.random.normal(0, 0.5, n_days)
    
    # Add some special events/holidays effect
    holiday_effect = np.zeros(n_days)
    # Add spikes around "holidays" (every 3 months)
    for i in range(0, n_days, 90):
        if i + 5 < n_days:
            holiday_effect[i:i+5] = 2.0
    
    # Combine all components
    y = trend + yearly_seasonality + weekly_seasonality + monthly_seasonality + holiday_effect + noise
    
    # Create DataFrame in Prophet format
    df = pd.DataFrame({
        'ds': dates,
        'y': y
    })
    
    print(f"Generated dataset shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    return df

def add_external_features(df):
    """
    Add external features that might influence the time series.
    These will be used as regressors in Prophet and for SHAP/LIME analysis.
    
    Args:
        df: Prophet DataFrame with 'ds' and 'y' columns
    
    Returns:
        DataFrame with additional features
    """
    print("Adding external features...")
    
    df = df.copy()
    
    # Date-based features
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['day_of_month'] = df['ds'].dt.day
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
    
    # Synthetic external variables
    n_days = len(df)
    
    # Temperature (seasonal pattern)
    df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + np.random.normal(0, 2, n_days)
    
    # Marketing spend (random with some correlation to sales)
    df['marketing_spend'] = np.maximum(0, 1000 + 500 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + 
                                      np.random.normal(0, 200, n_days))
    
    # Economic indicator (trending upward with noise)
    df['economic_index'] = 100 + 0.02 * np.arange(n_days) + np.random.normal(0, 5, n_days)
    
    # Competition activity (binary variable)
    df['competitor_active'] = np.random.binomial(1, 0.3, n_days)
    
    print(f"Added features: {[col for col in df.columns if col not in ['ds', 'y']]}")
    
    return df

# ============================================================================
# SECTION 2: Prophet Model Creation and Training
# ============================================================================

def build_and_train_prophet_model(df, external_features=None):
    """
    Build and train Prophet model with optional external regressors.
    
    Args:
        df: DataFrame with Prophet format and external features
        external_features: List of column names to use as external regressors
    
    Returns:
        model: Trained Prophet model
        forecast: Prophet forecast DataFrame
        test_df: Test data
    """
    print("Building and training Prophet model...")
    
    # Split data
    split_date = df['ds'].quantile(0.8)
    train_df = df[df['ds'] <= split_date].copy()
    test_df = df[df['ds'] > split_date].copy()
    
    print(f"Training data: {len(train_df)} days")
    print(f"Test data: {len(test_df)} days")
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    # Add external regressors if provided
    if external_features:
        for feature in external_features:
            model.add_regressor(feature)
            print(f"Added regressor: {feature}")
    
    # Fit the model
    print("Training Prophet model...")
    model.fit(train_df)
    
    # Make forecast for the entire period (including test)
    future = model.make_future_dataframe(periods=len(test_df))
    
    # Add external features to future DataFrame
    if external_features:
        for feature in external_features:
            future[feature] = df[feature].values
    
    forecast = model.predict(future)
    
    # Evaluate on test set
    test_forecast = forecast[forecast['ds'] > split_date]
    mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    return model, forecast, test_df, train_df

# ============================================================================
# SECTION 3: SHAP Explanations for Prophet
# ============================================================================

def create_prophet_wrapper_for_shap(model, feature_columns, base_df):
    """
    Create a wrapper function for Prophet that can be used with SHAP.
    
    Args:
        model: Trained Prophet model
        feature_columns: List of feature column names
        base_df: Base DataFrame with date information
    
    Returns:
        Wrapper function for Prophet predictions
    """
    def prophet_predict_wrapper(X):
        """
        Wrapper function that takes feature matrix and returns predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predictions array
        """
        predictions = []
        
        for i in range(len(X)):
            # Create a future DataFrame for this sample
            # Use a consistent date (first available date from base_df)
            future_df = pd.DataFrame({
                'ds': [base_df.iloc[0]['ds']]
            })
            
            # Add the features
            for j, feature in enumerate(feature_columns):
                future_df[feature] = X[i, j]
            
            # Make prediction
            pred = model.predict(future_df)
            predictions.append(pred['yhat'].iloc[0])
        
        return np.array(predictions)
    
    return prophet_predict_wrapper

def explain_prophet_with_shap(model, df, feature_columns, n_samples=100):
    """
    Apply SHAP KernelExplainer to Prophet model.
    
    Args:
        model: Trained Prophet model
        df: DataFrame with features
        feature_columns: List of feature column names
        n_samples: Number of samples to analyze
    
    Returns:
        shap_values: SHAP values
        X_sample: Feature matrix for analyzed samples
    """
    print("Applying SHAP KernelExplainer to Prophet...")
    
    # Prepare feature matrix
    X = df[feature_columns].values
    
    # Select samples to analyze
    sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sample = X[sample_indices]
    df_sample = df.iloc[sample_indices].copy()
    
    # Create Prophet wrapper
    prophet_wrapper = create_prophet_wrapper_for_shap(model, feature_columns, df_sample)
    
    # Create background dataset (smaller subset)
    background_size = min(50, len(X))
    background_indices = np.random.choice(len(X), background_size, replace=False)
    X_background = X[background_indices]
    
    # Create SHAP explainer
    print("Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(prophet_wrapper, X_background)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_sample[:20])  # Limit to 20 samples for demo
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    return shap_values, X_sample[:20], df_sample.iloc[:20]

def visualize_shap_prophet(shap_values, X_sample, feature_columns, df_sample):
    """
    Visualize SHAP values for Prophet model.
    
    Args:
        shap_values: SHAP values array
        X_sample: Feature matrix
        feature_columns: List of feature names
        df_sample: Sample DataFrame with dates
    """
    print("Creating SHAP visualizations for Prophet...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SHAP Analysis for Prophet Model', fontsize=16)
    
    # 1. Feature importance (mean absolute SHAP values)
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    axes[0, 0].bar(feature_columns, feature_importance)
    axes[0, 0].set_title('Feature Importance (Mean |SHAP Value|)')
    axes[0, 0].set_ylabel('Mean |SHAP Value|')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. SHAP values for first sample
    sample_idx = 0
    colors = ['red' if x < 0 else 'blue' for x in shap_values[sample_idx]]
    
    axes[0, 1].barh(feature_columns, shap_values[sample_idx], color=colors)
    axes[0, 1].set_title(f'SHAP Values for Sample {sample_idx}')
    axes[0, 1].set_xlabel('SHAP Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SHAP values distribution across samples
    axes[1, 0].boxplot([shap_values[:, i] for i in range(len(feature_columns))],
                      labels=feature_columns)
    axes[1, 0].set_title('SHAP Values Distribution')
    axes[1, 0].set_ylabel('SHAP Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature values vs SHAP values for most important feature
    most_important_idx = np.argmax(feature_importance)
    most_important_feature = feature_columns[most_important_idx]
    
    axes[1, 1].scatter(X_sample[:, most_important_idx], shap_values[:, most_important_idx], alpha=0.6)
    axes[1, 1].set_xlabel(f'{most_important_feature} Value')
    axes[1, 1].set_ylabel('SHAP Value')
    axes[1, 1].set_title(f'Feature Value vs SHAP Value - {most_important_feature}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/prophet_shap_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 4: LIME Explanations for Prophet
# ============================================================================

def explain_prophet_with_lime(model, df, feature_columns, sample_idx=0):
    """
    Apply LIME to explain Prophet predictions.
    
    Args:
        model: Trained Prophet model
        df: DataFrame with features
        feature_columns: List of feature column names
        sample_idx: Index of sample to explain
    
    Returns:
        explanation: LIME explanation object
    """
    print("Applying LIME to Prophet model...")
    
    # Prepare feature matrix
    X = df[feature_columns].values
    
    # Create Prophet wrapper for LIME
    def prophet_predict_wrapper_lime(X_batch):
        """Wrapper for Prophet predictions compatible with LIME."""
        predictions = []
        
        for i in range(len(X_batch)):
            # Create future DataFrame
            future_df = pd.DataFrame({
                'ds': [df.iloc[sample_idx]['ds']]  # Use the same date as the sample
            })
            
            # Add features
            for j, feature in enumerate(feature_columns):
                future_df[feature] = X_batch[i, j]
            
            # Make prediction
            pred = model.predict(future_df)
            predictions.append(pred['yhat'].iloc[0])
        
        return np.array(predictions)
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_columns,
        mode='regression',
        discretize_continuous=True,
        verbose=True
    )
    
    # Explain the specified instance
    instance = X[sample_idx]
    
    print(f"Explaining sample {sample_idx} (date: {df.iloc[sample_idx]['ds']})...")
    explanation = explainer.explain_instance(
        instance,
        prophet_predict_wrapper_lime,
        num_features=len(feature_columns),
        num_samples=500
    )
    
    return explanation

def visualize_lime_prophet(explanation, sample_idx, df, feature_columns):
    """
    Visualize LIME explanation for Prophet.
    
    Args:
        explanation: LIME explanation object
        sample_idx: Index of explained sample
        df: Original DataFrame
        feature_columns: List of feature names
    """
    print("Creating LIME visualizations for Prophet...")
    
    # Get feature importance from explanation
    exp_list = explanation.as_list()
    features, importances = zip(*exp_list)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(f'LIME Analysis for Prophet Model (Sample {sample_idx})', fontsize=16)
    
    # 1. Feature importance bar plot
    colors = ['red' if x < 0 else 'blue' for x in importances]
    y_pos = np.arange(len(features))
    
    axes[0].barh(y_pos, importances, color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(features)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Feature Contributions to Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (feature, importance) in enumerate(exp_list):
        axes[0].text(importance + 0.01 * max(importances), i, f'{importance:.3f}', 
                    va='center', fontsize=8)
    
    # 2. Explanation summary and sample details
    sample_date = df.iloc[sample_idx]['ds']
    actual_value = df.iloc[sample_idx]['y']
    predicted_value = explanation.predicted_value
    
    # Create text summary
    summary_text = f"""
Sample Details:
Date: {sample_date.strftime('%Y-%m-%d')}
Actual Value: {actual_value:.4f}
Predicted Value: {predicted_value:.4f}
Local Model R²: {explanation.score:.4f}

Feature Values:
"""
    
    # Add feature values
    for feature in feature_columns:
        value = df.iloc[sample_idx][feature]
        summary_text += f"{feature}: {value:.3f}\n"
    
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Sample Information')
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/prophet_lime_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed explanation to HTML
    explanation.save_to_file('/home/groot/MLOps/DEMO/data-drift-and-explainablity/lime_prophet_explanation.html')
    print("LIME explanation saved to: lime_prophet_explanation.html")

# ============================================================================
# SECTION 5: Prophet Model Visualization
# ============================================================================

def visualize_prophet_model(model, forecast, test_df, train_df):
    """
    Visualize Prophet model components and predictions.
    
    Args:
        model: Trained Prophet model
        forecast: Prophet forecast DataFrame
        test_df: Test data
        train_df: Training data
    """
    print("Creating Prophet model visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prophet Model Analysis', fontsize=16)
    
    # 1. Overall forecast vs actual
    axes[0, 0].plot(train_df['ds'], train_df['y'], 'b-', label='Training Data', alpha=0.7)
    axes[0, 0].plot(test_df['ds'], test_df['y'], 'g-', label='Actual Test', linewidth=2)
    
    # Get test forecast
    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
    axes[0, 0].plot(test_forecast['ds'], test_forecast['yhat'], 'r-', label='Predicted', linewidth=2)
    axes[0, 0].fill_between(test_forecast['ds'], test_forecast['yhat_lower'], 
                           test_forecast['yhat_upper'], alpha=0.3, color='red')
    
    axes[0, 0].set_title('Forecast vs Actual')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Trend component
    axes[0, 1].plot(forecast['ds'], forecast['trend'], 'purple', linewidth=2)
    axes[0, 1].set_title('Trend Component')
    axes[0, 1].set_ylabel('Trend')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Yearly seasonality
    axes[1, 0].plot(forecast['ds'], forecast['yearly'], 'orange', linewidth=2)
    axes[1, 0].set_title('Yearly Seasonality')
    axes[1, 0].set_ylabel('Yearly Effect')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Weekly seasonality
    axes[1, 1].plot(forecast['ds'], forecast['weekly'], 'green', linewidth=2)
    axes[1, 1].set_title('Weekly Seasonality')
    axes[1, 1].set_ylabel('Weekly Effect')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/prophet_model_components.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 6: Main Execution Function
# ============================================================================

def main():
    """
    Main function to demonstrate Prophet interpretability with SHAP and LIME.
    """
    print("=" * 80)
    print("Prophet Model Interpretability Demo")
    print("=" * 80)
    
    # Step 1: Generate data
    df = generate_synthetic_timeseries_for_prophet(1000)
    df = add_external_features(df)
    
    # Define external features to use as regressors
    external_features = ['temperature', 'marketing_spend', 'economic_index', 
                        'competitor_active', 'is_weekend']
    
    # Step 2: Train Prophet model
    model, forecast, test_df, train_df = build_and_train_prophet_model(df, external_features)
    
    # Step 3: Visualize Prophet model
    visualize_prophet_model(model, forecast, test_df, train_df)
    
    print("\n" + "=" * 50)
    print("SHAP Analysis")
    print("=" * 50)
    
    # Step 4: SHAP explanations
    shap_values, X_sample, df_sample = explain_prophet_with_shap(
        model, df, external_features, n_samples=50
    )
    visualize_shap_prophet(shap_values, X_sample, external_features, df_sample)
    
    print("\n" + "=" * 50)
    print("LIME Analysis")
    print("=" * 50)
    
    # Step 5: LIME explanations
    explanation = explain_prophet_with_lime(model, df, external_features, sample_idx=100)
    visualize_lime_prophet(explanation, sample_idx=100, df=df, feature_columns=external_features)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("Generated files:")
    print("- prophet_model_components.png")
    print("- prophet_shap_analysis.png")
    print("- prophet_lime_analysis.png")
    print("- lime_prophet_explanation.html")
    print("=" * 80)

if __name__ == "__main__":
    main()
