"""
Quick Demo: Basic SHAP and LIME Concepts
========================================

This script provides a simplified demonstration of SHAP and LIME concepts
using basic models and synthetic data. Perfect for understanding the fundamentals
before diving into the full LSTM and Prophet examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_dataset():
    """Create a simple synthetic dataset for demonstration."""
    np.random.seed(42)
    
    # Generate features
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    
    # Create target with known relationships
    # y = 2*x1 + 3*x2 - 1.5*x3 + 0.5*x4 + noise
    y = (2 * X[:, 0] + 
         3 * X[:, 1] - 
         1.5 * X[:, 2] + 
         0.5 * X[:, 3] + 
         0.1 * X[:, 4] + 
         np.random.normal(0, 0.1, n_samples))
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    return X, y, feature_names

def demonstrate_shap_basic():
    """Demonstrate SHAP with a simple Random Forest model."""
    print("=" * 60)
    print("SHAP Demonstration with Random Forest")
    print("=" * 60)
    
    # Create data
    X, y, feature_names = create_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model R² score: {model.score(X_test, y_test):.4f}")
    
    # SHAP explanation
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])  # Use first 100 samples
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SHAP Analysis - Random Forest Model', fontsize=16)
    
    # 1. Feature importance
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    axes[0, 0].bar(feature_names, feature_importance)
    axes[0, 0].set_title('Feature Importance (Mean |SHAP Value|)')
    axes[0, 0].set_ylabel('Mean |SHAP Value|')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. SHAP values for first sample
    sample_idx = 0
    colors = ['red' if x < 0 else 'blue' for x in shap_values[sample_idx]]
    axes[0, 1].barh(feature_names, shap_values[sample_idx], color=colors)
    axes[0, 1].set_title(f'SHAP Values for Sample {sample_idx}')
    axes[0, 1].set_xlabel('SHAP Value')
    
    # 3. SHAP values distribution
    axes[1, 0].boxplot([shap_values[:, i] for i in range(len(feature_names))],
                      labels=feature_names)
    axes[1, 0].set_title('SHAP Values Distribution')
    axes[1, 0].set_ylabel('SHAP Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Feature values vs SHAP values for most important feature
    most_important_idx = np.argmax(feature_importance)
    axes[1, 1].scatter(X_test[:100, most_important_idx], shap_values[:, most_important_idx], alpha=0.6)
    axes[1, 1].set_xlabel(f'{feature_names[most_important_idx]} Value')
    axes[1, 1].set_ylabel('SHAP Value')
    axes[1, 1].set_title(f'Feature Value vs SHAP Value\n{feature_names[most_important_idx]}')
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/basic_shap_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights
    print("\n📊 SHAP Insights:")
    print("-" * 30)
    for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    return model, X_test, y_test, feature_names, shap_values

def demonstrate_lime_basic(model, X_test, y_test, feature_names):
    """Demonstrate LIME with the same Random Forest model."""
    print("\n" + "=" * 60)
    print("LIME Demonstration with Random Forest")
    print("=" * 60)
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_test,
        feature_names=feature_names,
        mode='regression',
        verbose=True
    )
    
    # Explain a specific instance
    sample_idx = 0
    instance = X_test[sample_idx]
    
    print(f"Explaining sample {sample_idx}...")
    explanation = explainer.explain_instance(
        instance,
        model.predict,
        num_features=len(feature_names),
        num_samples=1000
    )
    
    # Get explanation data
    exp_list = explanation.as_list()
    features, importances = zip(*exp_list)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'LIME Analysis - Sample {sample_idx}', fontsize=16)
    
    # 1. Feature importance bar plot
    colors = ['red' if x < 0 else 'blue' for x in importances]
    axes[0].barh(features, importances, color=colors)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Feature Contributions to Prediction')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Explanation summary
    pred_value = explanation.predicted_value
    actual_value = y_test[sample_idx]
    
    summary_text = f"""
Prediction Details:
Predicted: {pred_value:.4f}
Actual: {actual_value:.4f}
Error: {abs(pred_value - actual_value):.4f}
Local R²: {explanation.score:.4f}

Feature Values:
"""
    
    for i, feature in enumerate(feature_names):
        value = instance[i]
        summary_text += f"{feature}: {value:.3f}\n"
    
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Prediction Summary')
    
    plt.tight_layout()
    plt.savefig('/home/groot/MLOps/DEMO/data-drift-and-explainablity/basic_lime_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed explanation
    explanation.save_to_file('/home/groot/MLOps/DEMO/data-drift-and-explainablity/basic_lime_explanation.html')
    
    # Print insights
    print("\n📊 LIME Insights:")
    print("-" * 30)
    print(f"Predicted value: {pred_value:.4f}")
    print(f"Actual value: {actual_value:.4f}")
    print(f"Local model R²: {explanation.score:.4f}")
    print("\nTop feature contributions:")
    for feature, importance in exp_list:
        sign = "↑" if importance > 0 else "↓"
        print(f"  {sign} {feature}: {importance:.4f}")
    
    return explanation

def compare_shap_lime():
    """Compare SHAP and LIME explanations."""
    print("\n" + "=" * 60)
    print("SHAP vs LIME Comparison")
    print("=" * 60)
    
    comparison_text = """
🔍 SHAP vs LIME - Key Differences:

📊 SCOPE:
• SHAP: Global + Local explanations
• LIME: Local explanations only

🎯 CONSISTENCY:
• SHAP: Mathematically consistent (additive property)
• LIME: Approximate local explanations

⚡ SPEED:
• SHAP: Fast for tree models, slower for complex models
• LIME: Consistently slower (perturbation-based)

🧮 METHOD:
• SHAP: Game theory (Shapley values)
• LIME: Local linear approximation

💯 ACCURACY:
• SHAP: Exact for tree models, approximate for others
• LIME: Always approximate

🎨 VISUALIZATION:
• SHAP: Rich built-in visualizations
• LIME: Basic plots + HTML reports

📈 USE CASES:
• SHAP: Model monitoring, global understanding
• LIME: Explaining individual predictions to stakeholders

🏭 PRODUCTION:
• SHAP: Better for batch processing
• LIME: Good for real-time individual explanations
"""
    
    print(comparison_text)

def main():
    """Main function to run the basic demonstration."""
    print("🚀 Basic SHAP and LIME Demonstration")
    print("=" * 60)
    print("This demo uses a simple Random Forest model to illustrate")
    print("the core concepts of SHAP and LIME interpretability methods.")
    print()
    
    # Run SHAP demonstration
    model, X_test, y_test, feature_names, shap_values = demonstrate_shap_basic()
    
    # Run LIME demonstration
    explanation = demonstrate_lime_basic(model, X_test, y_test, feature_names)
    
    # Compare methods
    compare_shap_lime()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Demo Complete!")
    print("=" * 60)
    print("Generated files:")
    print("• basic_shap_demo.png - SHAP visualizations")
    print("• basic_lime_demo.png - LIME visualizations")
    print("• basic_lime_explanation.html - Interactive LIME report")
    print()
    print("💡 Next Steps:")
    print("1. Review the generated visualizations")
    print("2. Open the HTML file for interactive exploration")
    print("3. Run the full LSTM and Prophet examples:")
    print("   python comprehensive_interpretability_demo.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
