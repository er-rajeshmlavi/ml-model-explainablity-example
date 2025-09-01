# SHAP and LIME Model Interpretability Examples

This repository contains comprehensive examples demonstrating how to use **SHAP** and **LIME** for model interpretability with LSTM and Prophet models in production environments.

## 📁 Project Structure

```
shap-lime-interpretability/
├── examples/                              # Example scripts and demos
│   ├── basic_shap_lime_demo.py           # Introduction to SHAP & LIME
│   ├── simple_shap_lime_examples.py      # Working LSTM & Prophet examples
│   ├── lstm_shap_lime_examples.py        # Detailed LSTM interpretability
│   └── prophet_shap_lime_examples.py     # Detailed Prophet interpretability
├── scripts/                              # Setup and utility scripts
│   ├── comprehensive_interpretability_demo.py  # Combined demo runner
│   └── setup_and_demo.py                # Installation and setup
|   ├── basic_shap_lime_demo.py           # Introduction to SHAP & LIME
│   ├── simple_shap_lime_examples.py      # Working LSTM & Prophet examples
│   ├── lstm_shap_lime_examples.py        # Detailed LSTM interpretability
│   └── prophet_shap_lime_examples.py     # Detailed Prophet interpretability
├── outputs/                              # Generated visualizations and reports
│   ├── *.png                            # Static visualizations
│   └── *.html                           # Interactive explanations
├── src/                                  # Reusable source code modules
├── tests/                                # Unit tests
├── docs/                                 # Documentation
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore rules
├── LICENSE                              # MIT License
├── CONTRIBUTING.md                      # Contribution guidelines
└── CHANGELOG.md                         # Version history
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Examples

#### Option A: Run All Examples (Recommended)

```bash
python scripts/comprehensive_interpretability_demo.py
```

#### Option B: Run Individual Examples

```bash
# Basic introduction to SHAP & LIME
python examples/basic_shap_lime_demo.py

# Simple working examples with LSTM & Prophet
python examples/simple_shap_lime_examples.py

# Detailed LSTM interpretability
python examples/lstm_shap_lime_examples.py

# Detailed Prophet interpretability
python examples/prophet_shap_lime_examples.py
```

## 📊 What You'll Get

### LSTM Model Interpretability
- **SHAP DeepExplainer** analysis showing:
  - Feature importance across time sequences
  - Temporal patterns in feature contributions
  - Global and local explanations
- **LIME TabularExplainer** providing:
  - Local explanations for individual predictions
  - Feature contribution breakdowns
  - Interactive HTML reports

### Prophet Model Interpretability  
- **SHAP KernelExplainer** analysis showing:
  - External regressor importance
  - Feature impact on forecasts
  - Global feature ranking
- **LIME TabularExplainer** providing:
  - Local explanations for specific forecasts
  - Feature contribution analysis
  - Detailed prediction breakdowns

## 🔍 Generated Outputs

After running the examples, you'll find these files:

### Visualizations (PNG)
- `lstm_shap_analysis.png` - SHAP analysis for LSTM model
- `lstm_lime_analysis.png` - LIME analysis for LSTM model  
- `prophet_model_components.png` - Prophet model components
- `prophet_shap_analysis.png` - SHAP analysis for Prophet model
- `prophet_lime_analysis.png` - LIME analysis for Prophet model

### Interactive Reports (HTML)
- `lime_lstm_explanation.html` - Detailed LSTM explanation
- `lime_prophet_explanation.html` - Detailed Prophet explanation

## 🛠 Key Features

### LSTM Examples
- **Synthetic Time Series Generation**: Creates realistic multi-feature time series data
- **LSTM Architecture**: 2-layer LSTM with dropout for sequence prediction
- **SHAP Integration**: 
  - Uses `DeepExplainer` for neural network interpretation
  - Analyzes temporal feature importance
  - Provides sequence-level explanations
- **LIME Integration**:
  - Adapts tabular explainer for sequence data
  - Explains individual time step predictions
  - Shows feature contributions over time

### Prophet Examples  
- **Time Series with External Regressors**: Synthetic data with seasonal patterns and external factors
- **Prophet Model**: Includes trend, seasonality, and external regressors
- **SHAP Integration**:
  - Uses `KernelExplainer` for Prophet's complex structure
  - Explains external regressor contributions
  - Provides global feature importance
- **LIME Integration**:
  - Explains individual forecast predictions
  - Shows how external factors affect forecasts
  - Provides local interpretability

## 📋 Model Details

### LSTM Model Architecture
```python
Sequential([
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False), 
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

### Prophet Model Configuration
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)
```

## 🔧 Customization

### For Your Own LSTM Models
1. Replace the synthetic data generation with your actual data
2. Modify the LSTM architecture to match your model
3. Adjust sequence length and features as needed
4. Update the SHAP background dataset size for performance

### For Your Own Prophet Models  
1. Replace synthetic data with your time series
2. Add your specific external regressors
3. Adjust Prophet parameters for your use case
4. Modify the feature engineering as needed

## 📚 Understanding the Results

### SHAP Values
- **Positive values**: Increase the prediction
- **Negative values**: Decrease the prediction  
- **Magnitude**: Indicates strength of impact
- **Sum property**: SHAP values sum to (prediction - baseline)

### LIME Explanations
- **Local focus**: Explains individual predictions only
- **Feature weights**: Show positive/negative contributions
- **Local model quality**: R² score indicates explanation reliability
- **Perturbation-based**: Uses local linear approximations

## 🚨 Performance Considerations

### SHAP
- **DeepExplainer**: Fast for neural networks, requires background data
- **KernelExplainer**: Model-agnostic but slower, good for complex models
- **Memory usage**: Scales with background dataset size

### LIME  
- **Sampling intensive**: Uses many perturbed samples
- **Local only**: Need to run for each prediction to explain
- **Computational cost**: Higher for real-time explanations

## 🔍 Production Recommendations

1. **Model Monitoring**: Use SHAP values to detect data drift
2. **Stakeholder Communication**: Use LIME for explaining specific predictions
3. **Batch Processing**: Pre-compute SHAP values for efficiency
4. **Documentation**: Keep interpretation methods documented for compliance
5. **Validation**: Cross-check explanations with domain expertise

## 🐛 Troubleshooting

### Common Issues
- **Memory errors**: Reduce background dataset size or sample size
- **Slow performance**: Use smaller sample sizes for demonstration
- **Import errors**: Ensure all packages are installed correctly
- **Prophet errors**: Try both `prophet` and `fbprophet` package names

### Dependencies
- TensorFlow ≥ 2.10.0 for LSTM models
- Prophet ≥ 1.1.0 for time series forecasting  
- SHAP ≥ 0.41.0 for model explanations
- LIME ≥ 0.2.0 for local interpretability

## 📖 Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## 📝 License

This project is provided as educational material for demonstrating ML interpretability techniques.
