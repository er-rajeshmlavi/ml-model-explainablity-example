"""
LSTM Model Interpretability with SHAP and LIME
===============================================

This script demonstrates how to use SHAP and LIME for interpreting LSTM model predictions.
We'll create a simple LSTM for time series prediction and apply both explainability methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shap
import lime
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# SECTION 1: Data Generation and Preprocessing
# ============================================================================

def generate_synthetic_time_series(n_samples=1000, n_features=5):
    """
    Generate synthetic time series data for demonstration.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features at each time step
    
    Returns:
        DataFrame with synthetic time series data
    """
    print("Generating synthetic time series data...")
    
    # Create time index
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate features with different patterns
    data = {}
    
    # Feature 1: Trend + seasonality
    trend = np.linspace(0, 10, n_samples)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    data['feature_1'] = trend + seasonal + np.random.normal(0, 0.5, n_samples)
    
    # Feature 2: Different seasonal pattern
    data['feature_2'] = 5 * np.cos(2 * np.pi * np.arange(n_samples) / 30) + np.random.normal(0, 0.3, n_samples)
    
    # Feature 3: Random walk
    data['feature_3'] = np.cumsum(np.random.normal(0, 0.1, n_samples))
    
    # Feature 4: Cyclical pattern
    data['feature_4'] = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 7) + np.random.normal(0, 0.2, n_samples)
    
    # Feature 5: Linear trend with noise
    data['feature_5'] = 0.01 * np.arange(n_samples) + np.random.normal(0, 0.1, n_samples)
    
    # Target: Combination of features with some non-linearity
    target = (0.3 * data['feature_1'] + 
              0.2 * data['feature_2'] + 
              0.1 * data['feature_3'] + 
              0.25 * data['feature_4'] + 
              0.15 * data['feature_5'] + 
              np.random.normal(0, 0.1, n_samples))
    
    df = pd.DataFrame(data, index=time_index)
    df['target'] = target
    
    print(f"Generated dataset shape: {df.shape}")
    return df

def create_sequences(data, sequence_length=10):
    """
    Create sequences for LSTM training.
    
    Args:
        data: Input data (numpy array)
        sequence_length: Length of input sequences
    
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# ============================================================================
# SECTION 2: LSTM Model Creation and Training
# ============================================================================

def build_lstm_model(input_shape):
    """
    Build and compile LSTM model.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
    
    Returns:
        Compiled LSTM model
    """
    print("Building LSTM model...")
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    print("Model architecture:")
    model.summary()
    return model

def train_lstm_model():
    """
    Complete pipeline to train LSTM model.
    
    Returns:
        model: Trained LSTM model
        X_test: Test sequences
        y_test: Test targets
        feature_names: Names of features
        scaler_X: Scaler for features
        scaler_y: Scaler for targets
    """
    # Generate data
    df = generate_synthetic_time_series(1000, 5)
    
    # Prepare features and target
    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    features = df[feature_cols].values
    target = df['target'].values
    
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = 10
    X, y = create_sequences(features_scaled, sequence_length)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build and train model
    model = build_lstm_model((sequence_length, len(feature_cols)))
    
    print("Training LSTM model...")
    history = model.fit(X_train, y_train, 
                       epochs=50, 
                       batch_size=32, 
                       validation_split=0.2, 
                       verbose=1)
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    
    return model, X_test, y_test, feature_cols, scaler_X, scaler_y

# ============================================================================
# SECTION 3: SHAP Explanations for LSTM
# ============================================================================

def explain_lstm_with_shap(model, X_test, feature_names, sample_size=100):
    """
    Apply SHAP DeepExplainer to LSTM model.
    
    Args:
        model: Trained LSTM model
        X_test: Test sequences
        feature_names: Names of features
        sample_size: Number of samples for explanation
    
    Returns:
        shap_values: SHAP values for the model
    """
    print("Applying SHAP DeepExplainer to LSTM...")
    
    # Select background data (subset of training data)
    background = X_test[:50]  # Use first 50 samples as background
    
    # Select samples to explain
    samples_to_explain = X_test[:sample_size]
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(samples_to_explain)
    
    # Handle the case where SHAP returns a list (for classification) vs array (for regression)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Take first element for regression
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # Reshape SHAP values for visualization
    # SHAP values shape: (n_samples, sequence_length, n_features)
    shap_values_reshaped = np.array(shap_values)
    
    return shap_values_reshaped, samples_to_explain

def visualize_shap_lstm(shap_values, X_samples, feature_names):
    """
    Visualize SHAP values for LSTM model.
    
    Args:
        shap_values: SHAP values
        X_samples: Input samples
        feature_names: Names of features
    """
    print("Creating SHAP visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SHAP Analysis for LSTM Model', fontsize=16)
    
    # 1. Summary plot for first sample
    sample_idx = 0
    shap_sample = shap_values[sample_idx]  # Shape: (sequence_length, n_features)
    
    # Average SHAP values across time steps for summary
    avg_shap = np.mean(np.abs(shap_sample), axis=0)
    
    axes[0, 0].bar(feature_names, avg_shap)
    axes[0, 0].set_title(f'Average Absolute SHAP Values (Sample {sample_idx})')
    axes[0, 0].set_ylabel('Mean |SHAP Value|')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. SHAP values over time for first feature
    feature_idx = 0
    time_steps = range(shap_sample.shape[0])
    
    axes[0, 1].plot(time_steps, shap_sample[:, feature_idx], marker='o')
    axes[0, 1].set_title(f'SHAP Values Over Time - {feature_names[feature_idx]}')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('SHAP Value')
    axes[0, 1].grid(True)
    
    # 3. Heatmap of SHAP values (time x features)
    im = axes[1, 0].imshow(shap_sample.T, aspect='auto', cmap='RdBu', 
                          vmin=-np.max(np.abs(shap_sample)), 
                          vmax=np.max(np.abs(shap_sample)))
    axes[1, 0].set_title('SHAP Values Heatmap (Features x Time)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Features')
    axes[1, 0].set_yticks(range(len(feature_names)))
    axes[1, 0].set_yticklabels(feature_names)
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. Feature importance across all samples
    all_sample_importance = np.mean(np.abs(shap_values), axis=(0, 1))
    
    axes[1, 1].bar(feature_names, all_sample_importance)
    axes[1, 1].set_title('Feature Importance (All Samples)')
    axes[1, 1].set_ylabel('Mean |SHAP Value|')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/lstm_shap_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 4: LIME Explanations for LSTM
# ============================================================================

def explain_lstm_with_lime(model, X_test, feature_names, sample_idx=0):
    """
    Apply LIME to explain LSTM predictions.
    
    Args:
        model: Trained LSTM model
        X_test: Test sequences
        feature_names: Names of features
        sample_idx: Index of sample to explain
    
    Returns:
        explanation: LIME explanation object
    """
    print("Applying LIME to LSTM model...")
    
    # For LSTM, we need to flatten the sequence data for LIME
    # Shape: (n_samples, sequence_length * n_features)
    X_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Create feature names for flattened data
    flat_feature_names = []
    for t in range(X_test.shape[1]):  # sequence_length
        for f in feature_names:
            flat_feature_names.append(f"{f}_t{t}")
    
    # Wrapper function for LIME
    def lstm_predict_wrapper(X_flat_batch):
        """Wrapper to convert flattened input back to LSTM format."""
        # Reshape back to LSTM format
        X_reshaped = X_flat_batch.reshape(-1, X_test.shape[1], X_test.shape[2])
        predictions = model.predict(X_reshaped)
        return predictions.flatten()
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_flat,
        feature_names=flat_feature_names,
        mode='regression',
        verbose=True
    )
    
    # Explain a single instance
    instance = X_flat[sample_idx]
    
    print(f"Explaining sample {sample_idx}...")
    explanation = explainer.explain_instance(
        instance, 
        lstm_predict_wrapper, 
        num_features=10,  # Show top 10 features
        num_samples=1000
    )
    
    return explanation, flat_feature_names

def visualize_lime_lstm(explanation, sample_idx):
    """
    Visualize LIME explanation for LSTM.
    
    Args:
        explanation: LIME explanation object
        sample_idx: Index of explained sample
    """
    print("Creating LIME visualizations...")
    
    # Get feature importance from explanation
    exp_list = explanation.as_list()
    features, importances = zip(*exp_list)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'LIME Analysis for LSTM Model (Sample {sample_idx})', fontsize=16)
    
    # 1. Feature importance bar plot
    colors = ['red' if x < 0 else 'blue' for x in importances]
    axes[0].barh(range(len(features)), importances, color=colors)
    axes[0].set_yticks(range(len(features)))
    axes[0].set_yticklabels([f.split('_')[0] + '_' + f.split('_')[1][-2:] for f in features], fontsize=8)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Top Features Contributing to Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Explanation summary
    explanation.save_to_file('/home/groot/MLOps/DEMO/data-drift-and-explainablity/lime_lstm_explanation.html')
    
    # Create a simple text summary plot
    pred_value = explanation.predicted_value
    axes[1].text(0.1, 0.8, f"Predicted Value: {pred_value:.4f}", fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.7, f"Local Model R²: {explanation.score:.4f}", fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.6, f"Features Analyzed: {len(features)}", fontsize=12, transform=axes[1].transAxes)
    
    # Show top 3 most important features
    axes[1].text(0.1, 0.4, "Top 3 Important Features:", fontsize=12, fontweight='bold', transform=axes[1].transAxes)
    for i, (feature, importance) in enumerate(exp_list[:3]):
        color = 'red' if importance < 0 else 'blue'
        axes[1].text(0.1, 0.3 - i*0.05, f"{i+1}. {feature.split('_')[0]}_{feature.split('_')[1][-2:]}: {importance:.4f}", 
                    fontsize=10, color=color, transform=axes[1].transAxes)
    
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Explanation Summary')
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/lstm_lime_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"LIME explanation saved to: lime_lstm_explanation.html")

# ============================================================================
# SECTION 5: Main Execution Function
# ============================================================================

def main():
    """
    Main function to demonstrate LSTM interpretability with SHAP and LIME.
    """
    print("=" * 80)
    print("LSTM Model Interpretability Demo")
    print("=" * 80)
    
    # Step 1: Train LSTM model
    model, X_test, y_test, feature_names, scaler_X, scaler_y = train_lstm_model()
    
    print("\n" + "=" * 50)
    print("SHAP Analysis")
    print("=" * 50)
    
    # Step 2: SHAP explanations
    shap_values, X_samples = explain_lstm_with_shap(model, X_test, feature_names, sample_size=50)
    visualize_shap_lstm(shap_values, X_samples, feature_names)
    
    print("\n" + "=" * 50)
    print("LIME Analysis")
    print("=" * 50)
    
    # Step 3: LIME explanations
    explanation, flat_feature_names = explain_lstm_with_lime(model, X_test, feature_names, sample_idx=0)
    visualize_lime_lstm(explanation, sample_idx=0)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("Generated files:")
    print("- lstm_shap_analysis.png")
    print("- lstm_lime_analysis.png") 
    print("- lime_lstm_explanation.html")
    print("=" * 80)

if __name__ == "__main__":
    main()
