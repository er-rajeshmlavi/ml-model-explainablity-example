"""
Simplified SHAP and LIME Examples for LSTM and Prophet
======================================================

This script provides working examples of SHAP and LIME with simplified implementations
that focus on demonstrating the core concepts without complex error handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import shap
import lime
from lime import lime_tabular
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def create_simple_lstm_example():
    """Create and demonstrate LSTM with SHAP and LIME."""
    print("=" * 60)
    print("SIMPLE LSTM EXAMPLE WITH SHAP AND LIME")
    print("=" * 60)
    
    # Generate simple time series data
    n_samples = 500
    n_features = 3
    sequence_length = 5
    
    # Create synthetic data
    time_steps = np.arange(n_samples)
    data = np.column_stack([
        np.sin(time_steps * 0.1) + np.random.normal(0, 0.1, n_samples),
        np.cos(time_steps * 0.05) + np.random.normal(0, 0.1, n_samples),
        time_steps * 0.01 + np.random.normal(0, 0.1, n_samples)
    ])
    
    # Create target (sum of features with some noise)
    target = np.sum(data, axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data_scaled = scaler_X.fit_transform(data)
    target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(target_scaled[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training LSTM on {len(X_train)} samples...")
    
    # Build simple LSTM model
    model = Sequential([
        LSTM(32, input_shape=(sequence_length, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Evaluate
    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    print(f"Test MAE: {mae:.4f}")
    
    print("\n" + "=" * 40)
    print("SHAP Analysis for LSTM")
    print("=" * 40)
    
    try:
        # SHAP explanation (simplified)
        background = X_test[:10]  # Small background set
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test[:20])
        
        # Handle list vs array return from SHAP
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Simple visualization
        feature_names = ['sin_feature', 'cos_feature', 'trend_feature']
        
        # Calculate average importance across time and samples
        avg_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(feature_names, avg_importance)
        plt.title('LSTM: Average Feature Importance (SHAP)')
        plt.ylabel('Mean |SHAP Value|')
        plt.xticks(rotation=45)
        
        # Show SHAP values for one sample
        sample_shap = shap_values[0]  # First sample
        plt.subplot(1, 2, 2)
        plt.imshow(sample_shap.T, aspect='auto', cmap='RdBu')
        plt.title('SHAP Values: Features x Time')
        plt.xlabel('Time Step')
        plt.ylabel('Features')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_lstm_shap.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ SHAP analysis completed for LSTM")
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    print("\n" + "=" * 40)
    print("LIME Analysis for LSTM")
    print("=" * 40)
    
    try:
        # LIME explanation (flatten sequences for tabular explainer)
        X_flat = X_test.reshape(X_test.shape[0], -1)
        flat_feature_names = [f"{feat}_t{t}" for t in range(sequence_length) 
                             for feat in feature_names]
        
        def lstm_predict_flat(X_flat_batch):
            X_reshaped = X_flat_batch.reshape(-1, sequence_length, n_features)
            return model.predict(X_reshaped).flatten()
        
        explainer = lime_tabular.LimeTabularExplainer(
            X_flat[:50],  # Smaller training set for LIME
            feature_names=flat_feature_names,
            mode='regression'
        )
        
        # Explain one instance
        explanation = explainer.explain_instance(
            X_flat[0], 
            lstm_predict_flat, 
            num_features=10
        )
        
        exp_list = explanation.as_list()
        features, importances = zip(*exp_list)
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if x < 0 else 'blue' for x in importances]
        plt.barh(range(len(features)), importances, color=colors)
        plt.yticks(range(len(features)), [f.split('_')[0]+'_'+f.split('_')[1][-2:] for f in features])
        plt.xlabel('Feature Importance')
        plt.title('LSTM: LIME Feature Contributions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_lstm_lime.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        explanation.save_to_file('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_lstm_lime.html')
        
        print("✓ LIME analysis completed for LSTM")
        
    except Exception as e:
        print(f"LIME analysis failed: {e}")

def create_simple_prophet_example():
    """Create and demonstrate Prophet with SHAP and LIME."""
    print("\n" + "=" * 60)
    print("SIMPLE PROPHET EXAMPLE WITH SHAP AND LIME")
    print("=" * 60)
    
    # Generate simple time series for Prophet
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # Simple synthetic data
    trend = np.linspace(10, 15, 365)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(365) / 365.25)
    noise = np.random.normal(0, 0.5, 365)
    y = trend + seasonal + noise
    
    # Create external features
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365.25) + np.random.normal(0, 2, 365)
    marketing = 100 + 50 * np.sin(2 * np.pi * np.arange(365) / 52) + np.random.normal(0, 10, 365)
    
    # Prophet DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'temperature': temperature,
        'marketing': marketing
    })
    
    print(f"Training Prophet on {len(df)} days...")
    
    # Train Prophet
    model = Prophet()
    model.add_regressor('temperature')
    model.add_regressor('marketing')
    
    # Split data
    train_df = df.iloc[:300].copy()
    test_df = df.iloc[300:].copy()
    
    model.fit(train_df)
    
    # Make predictions
    future = model.make_future_dataframe(periods=65)
    future['temperature'] = df['temperature'].values
    future['marketing'] = df['marketing'].values
    forecast = model.predict(future)
    
    # Evaluate
    test_forecast = forecast.iloc[300:]
    mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
    print(f"Test MAE: {mae:.4f}")
    
    print("\n" + "=" * 40)
    print("SHAP Analysis for Prophet")
    print("=" * 40)
    
    try:
        # Simplified SHAP for Prophet
        external_features = ['temperature', 'marketing']
        X = df[external_features].values
        
        def prophet_predict_simple(X_batch):
            predictions = []
            for i in range(len(X_batch)):
                future_single = pd.DataFrame({
                    'ds': [df.iloc[0]['ds']],  # Use fixed date
                    'temperature': [X_batch[i, 0]],
                    'marketing': [X_batch[i, 1]]
                })
                pred = model.predict(future_single)
                predictions.append(pred['yhat'].iloc[0])
            return np.array(predictions)
        
        # Small sample for demo
        sample_size = 50
        X_sample = X[:sample_size]
        
        explainer = shap.KernelExplainer(prophet_predict_simple, X_sample[:10])
        shap_values = explainer.shap_values(X_sample[:20])
        
        # Visualize
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(external_features, feature_importance)
        plt.title('Prophet: Feature Importance (SHAP)')
        plt.ylabel('Mean |SHAP Value|')
        
        plt.subplot(1, 2, 2)
        colors = ['red' if x < 0 else 'blue' for x in shap_values[0]]
        plt.barh(external_features, shap_values[0], color=colors)
        plt.title('SHAP Values for Sample 0')
        plt.xlabel('SHAP Value')
        
        plt.tight_layout()
        plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_prophet_shap.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ SHAP analysis completed for Prophet")
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    print("\n" + "=" * 40)
    print("LIME Analysis for Prophet")
    print("=" * 40)
    
    try:
        # LIME for Prophet
        explainer = lime_tabular.LimeTabularExplainer(
            X[:50],
            feature_names=external_features,
            mode='regression'
        )
        
        def prophet_predict_lime(X_batch):
            predictions = []
            for i in range(len(X_batch)):
                future_single = pd.DataFrame({
                    'ds': [df.iloc[0]['ds']],  # Use fixed date
                    'temperature': [X_batch[i, 0]],
                    'marketing': [X_batch[i, 1]]
                })
                pred = model.predict(future_single)
                predictions.append(pred['yhat'].iloc[0])
            return np.array(predictions)
        
        explanation = explainer.explain_instance(
            X[0], 
            prophet_predict_lime, 
            num_features=len(external_features)
        )
        
        exp_list = explanation.as_list()
        features, importances = zip(*exp_list)
        
        plt.figure(figsize=(8, 4))
        colors = ['red' if x < 0 else 'blue' for x in importances]
        plt.barh(features, importances, color=colors)
        plt.xlabel('Feature Importance')
        plt.title('Prophet: LIME Feature Contributions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_prophet_lime.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        explanation.save_to_file('/home/groot/MLOps/DEMO/data-drift-and-explainablity/simple_prophet_lime.html')
        
        print("✓ LIME analysis completed for Prophet")
        
    except Exception as e:
        print(f"LIME analysis failed: {e}")

def main():
    """Main function to run simplified examples."""
    print("🚀 Simplified SHAP & LIME Examples")
    print("=" * 60)
    print("This provides working examples of SHAP and LIME")
    print("with LSTM and Prophet models using simplified implementations.")
    print()
    
    # Run LSTM example
    create_simple_lstm_example()
    
    # Run Prophet example
    create_simple_prophet_example()
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print("=" * 60)
    print("Generated files:")
    print("• simple_lstm_shap.png")
    print("• simple_lstm_lime.png") 
    print("• simple_lstm_lime.html")
    print("• simple_prophet_shap.png")
    print("• simple_prophet_lime.png")
    print("• simple_prophet_lime.html")
    print()
    print("💡 Key Takeaways:")
    print("1. SHAP provides global feature importance + local explanations")
    print("2. LIME focuses on local explanations with model-agnostic approach")
    print("3. Both methods work with complex models like LSTM and Prophet")
    print("4. Visualizations help understand feature contributions")
    print("5. HTML reports provide interactive detailed explanations")
    print("=" * 60)

if __name__ == "__main__":
    main()
