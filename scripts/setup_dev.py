"""
Setup script for development environment.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print its status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def setup_development_environment():
    """Set up the development environment."""
    print("🚀 Setting up SHAP & LIME Interpretability Development Environment")
    print("=" * 70)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Not in a Git repository. Please run 'git init' first.")
        return False
    
    commands = [
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install pytest", "Installing test framework"),
        ("python -m pytest tests/ -v", "Running tests"),
        ("python examples/basic_shap_lime_demo.py", "Testing basic demo"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
        print()  # Add spacing
    
    print("=" * 70)
    print(f"✅ Setup completed: {success_count}/{len(commands)} steps successful")
    
    if success_count == len(commands):
        print("🎉 Development environment is ready!")
        print("\n📚 Next steps:")
        print("1. Run examples: python examples/simple_shap_lime_examples.py")
        print("2. Explore the code in examples/ directory")
        print("3. Check outputs/ for generated visualizations")
        print("4. Read CONTRIBUTING.md for development guidelines")
    else:
        print("⚠️  Some setup steps failed. Please check the errors above.")
    
    return success_count == len(commands)


if __name__ == "__main__":
    setup_development_environment()
