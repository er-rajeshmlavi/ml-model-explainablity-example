"""
Installation and Setup Script for SHAP & LIME Examples
======================================================

This script ensures all dependencies are properly installed and provides
a working demo of SHAP and LIME interpretability methods.
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    required_packages = {
        'tensorflow': 'tensorflow>=2.10.0,<2.17.0',
        'prophet': 'prophet>=1.1.0',
        'shap': 'shap>=0.41.0,<0.46.0',
        'lime': 'lime>=0.2.0',
        'numpy': 'numpy>=1.21.0,<2.0.0',
        'pandas': 'pandas>=1.3.0',
        'matplotlib': 'matplotlib>=3.5.0,<4.0.0',
        'seaborn': 'seaborn>=0.11.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'pytz': 'pytz>=2021.1',
        'python-dateutil': 'python-dateutil>=2.8.0'
    }
    
    missing_packages = []
    
    print("Checking dependencies...")
    for package, requirement in required_packages.items():
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(requirement)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                install_package(package)
                print(f"✓ Installed {package}")
            except Exception as e:
                print(f"✗ Failed to install {package}: {e}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def run_examples():
    """Run the working examples."""
    print("\n" + "=" * 60)
    print("RUNNING SHAP & LIME EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Basic SHAP & LIME Demo", "basic_shap_lime_demo.py"),
        ("Simple LSTM & Prophet Examples", "simple_shap_lime_examples.py")
    ]
    
    for name, script in examples:
        print(f"\n--- {name} ---")
        try:
            result = subprocess.run([sys.executable, script], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ {script} completed successfully")
                # Print last few lines of output
                output_lines = result.stdout.split('\n')[-10:]
                for line in output_lines:
                    if line.strip():
                        print(f"  {line}")
            else:
                print(f"✗ {script} failed with error:")
                print(result.stderr)
        except subprocess.TimeoutExpired:
            print(f"⚠ {script} timed out (>5 minutes)")
        except Exception as e:
            print(f"✗ Error running {script}: {e}")

def main():
    """Main setup and demo function."""
    print("🚀 SHAP & LIME Model Interpretability Setup")
    print("=" * 60)
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("⚠ Some dependencies failed to install. Continuing anyway...")
    
    # Run examples
    run_examples()
    
    print("\n" + "=" * 60)
    print("🎯 SETUP COMPLETE")
    print("=" * 60)
    print("Available examples:")
    print("1. basic_shap_lime_demo.py - Introduction to SHAP & LIME concepts")
    print("2. simple_shap_lime_examples.py - LSTM & Prophet working examples")
    print("3. lstm_shap_lime_examples.py - Detailed LSTM interpretability")
    print("4. prophet_shap_lime_examples.py - Detailed Prophet interpretability")
    print("5. comprehensive_interpretability_demo.py - Complete demo runner")
    
    print("\n📁 Generated Files:")
    import os
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    
    for file in png_files:
        print(f"  📊 {file}")
    for file in html_files:
        print(f"  🌐 {file}")
    
    print("\n💡 Next Steps:")
    print("1. View PNG files for visual insights")
    print("2. Open HTML files in browser for interactive explanations")
    print("3. Adapt the code for your own models and data")
    print("4. Integrate interpretability into your ML pipeline")
    print("=" * 60)

if __name__ == "__main__":
    main()
