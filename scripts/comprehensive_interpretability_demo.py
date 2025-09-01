"""
Comprehensive Example: SHAP and LIME for ML Model Interpretability
==================================================================

This script demonstrates both LSTM and Prophet interpretability in a single workflow.
It's designed to be a comprehensive tutorial showing best practices for applying
SHAP and LIME to production ML models.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import importlib

# Import our example modules
try:
    # Add current directory to path
    # Add the parent directory of this script to sys.path for module imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    
    # Import our modules
    import lstm_shap_lime_examples as lstm_module
    import prophet_shap_lime_examples as prophet_module
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        'tensorflow', 'prophet', 'shap', 'lime', 
        'numpy', 'pandas', 'matplotlib', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def run_lstm_example():
    """
    Run the LSTM interpretability example.
    """
    print("\n" + "=" * 80)
    print("RUNNING LSTM INTERPRETABILITY EXAMPLE")
    print("=" * 80)
    
    try:
        lstm_module.main()
        print("\n✓ LSTM example completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error running LSTM example: {e}")
        return False

def run_prophet_example():
    """
    Run the Prophet interpretability example.
    """
    print("\n" + "=" * 80)
    print("RUNNING PROPHET INTERPRETABILITY EXAMPLE")
    print("=" * 80)
    
    try:
        prophet_module.main()
        print("\n✓ Prophet example completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error running Prophet example: {e}")
        return False

def create_summary_report():
    """
    Create a summary report of all generated files and analyses.
    """
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    expected_files = [
        'lstm_shap_analysis.png',
        'lstm_lime_analysis.png',
        'lime_lstm_explanation.html',
        'prophet_model_components.png',
        'prophet_shap_analysis.png',
        'prophet_lime_analysis.png',
        'lime_prophet_explanation.html'
    ]
    
    existing_files = []
    missing_files = []
    
    base_path = '/home/groot/MLOps/DEMO/data-drift-and-explainablity'
    
    for file in expected_files:
        full_path = os.path.join(base_path, file)
        if os.path.exists(full_path):
            existing_files.append(file)
            print(f"✓ Generated: {file}")
        else:
            missing_files.append(file)
            print(f"✗ Missing: {file}")
    
    print(f"\nGenerated files: {len(existing_files)}/{len(expected_files)}")
    
    if existing_files:
        print("\n📊 ANALYSIS OUTPUTS:")
        print("-" * 40)
        
        # LSTM outputs
        lstm_files = [f for f in existing_files if 'lstm' in f.lower()]
        if lstm_files:
            print("LSTM Model Interpretability:")
            for file in lstm_files:
                if 'shap' in file:
                    print(f"  • {file} - SHAP feature importance and temporal analysis")
                elif 'lime' in file:
                    print(f"  • {file} - LIME local explanations")
        
        # Prophet outputs
        prophet_files = [f for f in existing_files if 'prophet' in f.lower()]
        if prophet_files:
            print("\nProphet Model Interpretability:")
            for file in prophet_files:
                if 'component' in file:
                    print(f"  • {file} - Prophet model components visualization")
                elif 'shap' in file:
                    print(f"  • {file} - SHAP feature importance for time series")
                elif 'lime' in file:
                    print(f"  • {file} - LIME explanations for forecasts")
    
    return len(existing_files) == len(expected_files)

def print_usage_guide():
    """
    Print a guide on how to interpret the results.
    """
    guide = """
📖 INTERPRETATION GUIDE
========================

🔍 SHAP (SHapley Additive exPlanations):
----------------------------------------
• Global interpretability: Shows feature importance across all predictions
• Local interpretability: Explains individual predictions
• Additive property: Sum of SHAP values = prediction - baseline
• Positive SHAP values increase prediction, negative values decrease it

Key Visualizations:
• Summary plots: Feature importance ranking
• Waterfall plots: Step-by-step contribution breakdown
• Force plots: Visual representation of feature contributions
• Dependence plots: How feature values affect predictions

📊 LIME (Local Interpretable Model-agnostic Explanations):
---------------------------------------------------------
• Local interpretability only: Explains individual predictions
• Model-agnostic: Works with any ML model
• Creates simple interpretable models locally
• Perturbs input features to understand their impact

Key Visualizations:
• Feature contribution bars: Positive/negative impact on prediction
• Text/HTML explanations: Detailed breakdown of local model
• Feature value displays: Shows actual values used in explanation

🔬 MODEL-SPECIFIC CONSIDERATIONS:
=================================

LSTM Models:
• Time dimension adds complexity to interpretability
• SHAP values show temporal feature importance
• Consider sequence-level vs. step-level explanations
• DeepExplainer works well for neural networks

Prophet Models:
• Interpretability focuses on external regressors
• Built-in component decomposition (trend, seasonality)
• SHAP/LIME explain regressor contributions
• KernelExplainer works well for Prophet's complex structure

💡 PRODUCTION RECOMMENDATIONS:
==============================
1. Use SHAP for global model understanding and monitoring
2. Use LIME for explaining specific predictions to stakeholders
3. Monitor SHAP values for data drift detection
4. Combine explanations with domain expertise
5. Document interpretation methods for model governance
6. Consider computational costs for real-time explanations

📁 FILES STRUCTURE:
==================
• PNG files: Static visualizations for reports
• HTML files: Interactive explanations for detailed analysis
• Python scripts: Reusable code for your models
"""
    print(guide)

def main():
    """
    Main function to run the comprehensive interpretability demo.
    """
    print("🚀 SHAP & LIME Model Interpretability Comprehensive Demo")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Ask user which examples to run
    print("\nSelect which examples to run:")
    print("1. LSTM only")
    print("2. Prophet only") 
    print("3. Both (recommended)")
    
    try:
        choice = input("\nEnter your choice (1-3) [default: 3]: ").strip()
        if not choice:
            choice = "3"
    except KeyboardInterrupt:
        print("\nDemo cancelled by user.")
        return
    
    success_count = 0
    total_examples = 0
    
    # Run selected examples
    if choice in ["1", "3"]:
        total_examples += 1
        if run_lstm_example():
            success_count += 1
    
    if choice in ["2", "3"]:
        total_examples += 1
        if run_prophet_example():
            success_count += 1
    
    # Create summary report
    all_files_generated = create_summary_report()
    
    # Print usage guide
    print_usage_guide()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 DEMO COMPLETION SUMMARY")
    print("=" * 80)
    print(f"Examples run: {success_count}/{total_examples}")
    print(f"Files generated: {'✓ All files created' if all_files_generated else '⚠ Some files missing'}")
    
    if success_count == total_examples and all_files_generated:
        print("\n🎉 Demo completed successfully!")
        print("You can now explore the generated visualizations and HTML reports.")
    else:
        print("\n⚠ Demo completed with some issues.")
        print("Check the error messages above for troubleshooting.")
    
    print("\n📚 Next steps:")
    print("1. Review the generated PNG files for visual insights")
    print("2. Open HTML files in a browser for interactive explanations")
    print("3. Adapt the code for your own models and datasets")
    print("4. Integrate interpretability into your ML pipeline")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
