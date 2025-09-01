# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository structure and organization
- Comprehensive examples for SHAP and LIME interpretability
- Support for LSTM and Prophet model explanations

## [1.0.0] - 2025-09-01

### Added
- Basic SHAP and LIME demonstration with Random Forest
- Simple LSTM interpretability examples with SHAP DeepExplainer and LIME
- Prophet model interpretability with external regressors
- Working examples with synthetic data generation
- Comprehensive visualization outputs (PNG and HTML)
- Proper error handling and dependency management
- Documentation and setup scripts

### Features
- **LSTM Model Interpretability**:
  - SHAP DeepExplainer for temporal feature analysis
  - LIME TabularExplainer adapted for sequence data
  - Feature importance visualization over time sequences
  
- **Prophet Model Interpretability**:
  - SHAP KernelExplainer for external regressor analysis
  - LIME explanations for forecast predictions
  - Model component visualization (trend, seasonality)
  
- **General Features**:
  - Self-contained examples with synthetic data
  - Multiple difficulty levels (basic → simple → comprehensive)
  - Interactive HTML reports for detailed analysis
  - Proper Git repository structure

### Technical Details
- Python 3.8+ compatibility
- TensorFlow/Keras for LSTM models
- Prophet for time series forecasting
- SHAP ≥0.41.0 for model explanations
- LIME ≥0.2.0 for local interpretability
- Matplotlib/Seaborn for visualizations

### Documentation
- Comprehensive README with usage instructions
- Contributing guidelines
- Installation and setup documentation
- Code examples and best practices

### Files Structure
```
├── examples/          # Example scripts
├── scripts/           # Setup and demo runners
├── outputs/           # Generated visualizations
├── docs/              # Documentation
├── tests/             # Test files
└── src/               # Source code modules
```

### Known Issues
- LSTM SHAP integration requires careful handling of return types
- Prophet wrapper functions need consistent date handling
- Some examples may require significant computational time
- Memory usage can be high for large sample sizes

### Dependencies
- tensorflow>=2.10.0,<2.17.0
- prophet>=1.1.0
- shap>=0.41.0,<0.46.0
- lime>=0.2.0
- numpy>=1.21.0,<2.0.0
- pandas>=1.3.0
- matplotlib>=3.5.0,<4.0.0
- scikit-learn>=1.0.0
