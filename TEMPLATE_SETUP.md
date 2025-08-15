# Template Setup Guide

This guide will help you customize this template for your specific Python project.

## 🚀 Quick Setup

### 1. Clone or Use Template
```bash
# Option A: Use GitHub template (recommended)
# Click "Use this template" button on GitHub

# Option B: Clone repository
git clone <repository-url> my-new-project
cd my-new-project
```

### 2. Customize Project Information

#### Update `pyproject.toml`
```toml
[project]
name = "your-project-name"           # Change this
description = "Your project description"  # Change this
# Update other fields as needed
```

#### Update package name
```bash
# Rename the source directory
mv src/python_uv_template src/your_project_name

# Update the PACKAGE_NAME in Makefile
# Change: PACKAGE_NAME := python_uv_template
# To:     PACKAGE_NAME := your_project_name
```

### 3. Update Import References

#### In `tests/test_main.py`
```python
# Change this line:
from python_uv_template.main import greet, process_names, main

# To:
from your_project_name.main import greet, process_names, main
```

#### In `pyproject.toml` (if using entry points)
```toml
[project.scripts]
main = "your_project_name.main:main"  # Update this
```

### 4. Update Tool Configurations

#### MyPy configuration in `pyproject.toml`
```toml
[tool.isort]
known_first_party = ["your_project_name"]  # Update this
```

### 5. Install and Test
```bash
# Install development dependencies
make dev-install

# Run all checks to ensure everything works
make all-checks

# Test the application
make run
```

## 🔧 Customization Options

### Python Version
To change the Python version requirement:

1. Update `pyproject.toml`:
```toml
requires-python = ">=3.11"  # Change as needed
```

2. Update `.github/workflows/ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12", "3.13"]  # Add versions as needed
```

3. Update tool configurations:
```toml
[tool.ruff]
target-version = "py311"  # Update accordingly

[tool.black]
target-version = ['py311']  # Update accordingly

[tool.mypy]
python_version = "3.11"  # Update accordingly
```

### Dependencies
Add your project dependencies to `pyproject.toml`:

```toml
[project]
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0.0",
    # Add your dependencies here
]

[dependency-groups]
dev = [
    # Keep existing dev dependencies
    "mypy>=1.17.1",
    "pytest>=8.4.1",
    # ... existing dev deps

    # Add additional dev dependencies if needed
    "jupyter>=1.0.0",
    "ipython>=8.0.0",
]
```

### Code Quality Rules
Customize linting and formatting rules in `pyproject.toml`:

#### Ruff Configuration
```toml
[tool.ruff]
# Add or remove rules as needed
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
    "N",  # pep8-naming
    "S",  # bandit
    "D",  # pydocstyle (add if you want docstring checks)
]

# Customize ignored rules
ignore = [
    "E501",  # line too long, handled by black
    "D100",  # Missing docstring in public module (if using D)
]
```

#### MyPy Configuration
```toml
[tool.mypy]
# Adjust strictness as needed
strict = true  # Enable all strict checks
# Or configure individual checks:
# disallow_untyped_defs = true
# disallow_incomplete_defs = true
```

### Pre-commit Hooks
Customize `.pre-commit-config.yaml` to add or remove hooks:

```yaml
repos:
  # Add additional hooks
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.3
    hooks:
      - id: prettier
        types_or: [javascript, jsx, ts, tsx, json, yaml, markdown]
```

### GitHub Actions
Customize `.github/workflows/ci.yml`:

```yaml
# Add additional jobs
jobs:
  test:
    # ... existing test job

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

## 📁 Project Structure Recommendations

### For CLI Applications
```
src/your_project_name/
├── __init__.py
├── main.py          # Entry point
├── cli/             # CLI-specific code
│   ├── __init__.py
│   └── commands.py
├── core/            # Business logic
│   ├── __init__.py
│   └── logic.py
└── utils/           # Utilities
    ├── __init__.py
    └── helpers.py
```

### For Libraries
```
src/your_project_name/
├── __init__.py      # Public API
├── core.py          # Main functionality
├── exceptions.py    # Custom exceptions
├── types.py         # Type definitions
└── utils.py         # Utilities
```

### For Web Applications
```
src/your_project_name/
├── __init__.py
├── app.py           # Application factory
├── models/          # Data models
├── views/           # View functions/classes
├── services/        # Business logic
└── utils/           # Utilities
```

## 🧪 Testing Structure

### Test Organization
```
tests/
├── __init__.py
├── conftest.py      # Pytest fixtures
├── unit/            # Unit tests
│   ├── __init__.py
│   └── test_core.py
├── integration/     # Integration tests
│   ├── __init__.py
│   └── test_api.py
└── fixtures/        # Test data
    └── sample_data.json
```

### Common Test Patterns
```python
# conftest.py
import pytest
from your_project_name import create_app

@pytest.fixture
def app():
    """Create application for testing."""
    return create_app(testing=True)

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()
```

## 🚀 Deployment Considerations

### Docker Support
Add `Dockerfile`:
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/

# Set Python path
ENV PYTHONPATH=/app/src

# Run application
CMD ["uv", "run", "python", "-m", "your_project_name.main"]
```

### Environment Variables
Create `.env.example`:
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# API Keys
API_KEY=your_api_key_here

# Environment
ENVIRONMENT=development
DEBUG=true
```

## 📝 Documentation

### Add Sphinx Documentation
```bash
# Add to dev dependencies in pyproject.toml
[dependency-groups]
dev = [
    # ... existing deps
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
]

# Initialize docs
mkdir docs
cd docs
sphinx-quickstart
```

### API Documentation
For libraries, consider adding docstring examples:
```python
def your_function(param: str) -> str:
    """
    Brief description of the function.

    Args:
        param: Description of the parameter

    Returns:
        Description of the return value

    Raises:
        ValueError: When param is invalid

    Example:
        >>> your_function("test")
        'processed: test'
    """
    return f"processed: {param}"
```

## 🔄 Maintenance

### Regular Updates
```bash
# Update dependencies
make update-deps

# Update pre-commit hooks
make pre-commit-update

# Check for security vulnerabilities
make audit
```

### Version Management
Consider using semantic versioning and tools like `bump2version`:
```bash
# Add to dev dependencies
pip install bump2version

# Create .bumpversion.cfg
[bumpversion]
current_version = 0.1.0
commit = True
tag = True

[bumpversion:file:pyproject.toml]
[bumpversion:file:src/your_project_name/__init__.py]
```

This template provides a solid foundation for any Python project. Customize it according to your specific needs and requirements!
