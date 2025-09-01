## Contributing to SHAP & LIME Model Interpretability Examples

We welcome contributions to improve and extend these interpretability examples!

### 🤝 How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### 📋 Contribution Guidelines

#### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

#### Documentation
- Update README.md if adding new examples
- Add inline comments for complex algorithms
- Include usage examples in docstrings
- Update requirements.txt for new dependencies

#### Testing
- Add unit tests for new functions
- Ensure all examples run without errors
- Test with different Python versions (3.8+)
- Verify outputs are generated correctly

### 🐛 Bug Reports

When filing an issue, please include:
- Python version and OS
- Complete error message and stack trace
- Minimal code example to reproduce the issue
- Expected vs actual behavior

### 💡 Feature Requests

For new features, please:
- Explain the use case and motivation
- Provide examples of how it would be used
- Consider backwards compatibility
- Check if similar functionality already exists

### 🔧 Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/shap-lime-interpretability.git
cd shap-lime-interpretability
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

4. Run tests:
```bash
python -m pytest tests/
```

### 📝 Pull Request Process

1. Ensure your code passes all tests
2. Update documentation as needed
3. Add your changes to CHANGELOG.md
4. Request review from maintainers
5. Address any feedback promptly

### 🏷️ Versioning

We use semantic versioning (SemVer). Version numbers follow the pattern:
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

### 📞 Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Join discussions in the Issues section
- Review the documentation in the `docs/` directory

### 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Special recognition for major improvements

Thank you for contributing to making ML interpretability more accessible! 🚀
