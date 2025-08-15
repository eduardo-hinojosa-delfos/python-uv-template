# Python UV Template

A comprehensive Python project template using `uv` as package manager, with all code quality tools pre-configured and ready to use.

## 🚀 Features

- **Dependency Management**: `uv` for fast package installation and virtual environment management
- **Code Quality**: Ruff, Black, isort, MyPy pre-configured
- **Security**: Bandit for security analysis, Safety for dependency auditing
- **Testing**: pytest with coverage reporting and parallel execution
- **Pre-commit Hooks**: Automatic code quality checks before each commit
- **CI/CD**: GitHub Actions workflow ready to use
- **Makefile**: Quick commands for all development tasks
- **Modern Python**: Configured for Python 3.13+ with latest best practices

## 📋 Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) installed

## 🛠️ Quick Start

1. Use this template or clone the repository:
```bash
git clone <your-repo>
cd python-uv-template
```

2. Install development dependencies:
```bash
make dev-install
```

3. Start coding in `src/` and add tests in `tests/`

4. Run quality checks:
```bash
make check
```

## 🎯 Available Commands

### Installation & Setup
```bash
make install          # Install production dependencies
make dev-install      # Install development dependencies
make clean           # Clean temporary files and cache
```

### Testing
```bash
make test            # Run tests
make test-cov        # Run tests with coverage report
make test-parallel   # Run tests in parallel
```

### Code Quality
```bash
make lint            # Run linting with ruff
make lint-fix        # Run linting with auto-fix
make format          # Format code (black + isort)
make format-check    # Check formatting without modifying files
make type-check      # Type checking with mypy
```

### Security
```bash
make security        # Security analysis with bandit
make audit          # Audit dependencies with safety
```

### Pre-commit
```bash
make pre-commit      # Run pre-commit on all files
make pre-commit-update # Update pre-commit hooks
```

### Comprehensive Checks
```bash
make check          # Basic checks (format, lint, types, security)
make all-checks     # Complete suite (includes tests and audit)
make ci            # CI/CD checks
```

### Development
```bash
make run           # Run the main application
make build         # Build the package
make env-info      # Show environment information
make update-deps   # Update dependencies
```

### Docker
```bash
make docker-build      # Build production Docker image
make docker-build-dev  # Build development Docker image
make docker-run        # Run production container
make docker-run-dev    # Run development container
make docker-compose-up # Start with docker-compose
make docker-compose-dev # Start development environment
make docker-clean      # Clean Docker resources
```

## 🔧 Configured Tools

### Ruff
- Ultra-fast Python linter and formatter
- Configured with pycodestyle, pyflakes, isort, bandit rules
- Auto-fix capabilities for common issues

### Black
- Uncompromising code formatter
- 88 character line length
- Python 3.13+ compatible

### isort
- Automatic import sorting
- Black-compatible profile
- Consistent import organization

### MyPy
- Static type checking
- Strict configuration for better code quality
- Python 3.13 support

### Bandit
- Security vulnerability scanner
- Common security issue detection
- Customizable security rules

### Safety
- Modern dependency vulnerability auditing with `scan` command
- Known security vulnerability detection
- Regular security database updates

### pytest
- Modern testing framework
- Built-in coverage reporting
- Parallel test execution support

### Pre-commit
- Automatic hooks before commits
- Integration with all quality tools
- Ready-to-use configuration

## 📁 Project Structure

```
.
├── .github/workflows/     # GitHub Actions CI/CD
├── deploy/               # Docker deployment files
│   ├── Dockerfile        # Production Docker image
│   ├── Dockerfile.dev    # Development Docker image
│   ├── docker-compose.yml # Production deployment
│   └── docker-compose.dev.yml # Development environment
├── src/python_uv_template/  # Source code (rename to your project)
├── tests/                 # Test files
├── .pre-commit-config.yaml # Pre-commit configuration
├── Makefile              # Development automation commands
├── pyproject.toml        # Project configuration and dependencies
└── README.md            # This file
```

## 🚦 Development Workflow

1. **Development**: Write your code in `src/`
2. **Testing**: Add tests in `tests/`
3. **Quality Check**: Run `make check` before committing
4. **Commit**: Pre-commit hooks run automatically
5. **CI/CD**: GitHub Actions runs the complete test suite

## 📊 Coverage Reports

Coverage reports are generated in:
- **Terminal**: Summary with missing lines
- **HTML**: `htmlcov/index.html` for detailed view

## 🎯 Using This Template

1. **Click "Use this template"** or clone the repository
2. **Rename the package**: Change `src/python_uv_template/` to your project name
3. **Update `pyproject.toml`**: Change project name, description, and author
4. **Update imports**: Replace `python_uv_template` imports in tests and code
5. **Customize**: Modify the example code to fit your needs
6. **Start developing**: Run `make dev-install` and begin coding!

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run `make all-checks` to ensure code quality
4. Commit your changes (`git commit -am 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🆘 Help

To see all available commands:
```bash
make help
```

For environment information:
```bash
make env-info
```

## 🌟 Why This Template?

- **Modern Python**: Uses the latest Python 3.13+ features
- **Fast Development**: `uv` provides lightning-fast dependency resolution
- **Quality First**: All major code quality tools pre-configured
- **Security Focused**: Built-in security scanning and dependency auditing
- **CI/CD Ready**: GitHub Actions workflow included
- **Docker Ready**: Production and development Docker configurations
- **Developer Friendly**: Comprehensive Makefile for common tasks
- **Best Practices**: Follows Python packaging and development best practices
