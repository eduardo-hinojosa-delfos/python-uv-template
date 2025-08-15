.PHONY: help install dev-install clean test test-cov lint format check security audit pre-commit all-checks ci
.DEFAULT_GOAL := help

# Variables
PYTHON := python
UV := uv
SRC_DIR := src
TEST_DIR := tests
PACKAGE_NAME := python_uv_template

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(UV) sync --no-dev

dev-install: ## Install development dependencies
	$(UV) sync
	$(UV) run pre-commit install

clean: ## Clean temporary files and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Testing
test: ## Run tests
	$(UV) run pytest $(TEST_DIR) -v

test-cov: ## Run tests with coverage
	$(UV) run pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v

test-parallel: ## Run tests in parallel
	$(UV) run pytest $(TEST_DIR) -n auto -v

# Code Quality
lint: ## Run linting with ruff
	$(UV) run ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run linting with ruff and auto-fix
	$(UV) run ruff check $(SRC_DIR) $(TEST_DIR) --fix

format: ## Format code with black and isort
	$(UV) run black $(SRC_DIR) $(TEST_DIR)
	$(UV) run isort $(SRC_DIR) $(TEST_DIR)

format-check: ## Check formatting without modifying files
	$(UV) run black --check $(SRC_DIR) $(TEST_DIR)
	$(UV) run isort --check-only $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking with mypy
	$(UV) run mypy $(SRC_DIR)

# Security
security: ## Run security analysis with bandit
	$(UV) run bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	$(UV) run bandit -r $(SRC_DIR)

audit: ## Audit dependencies with safety
	$(UV) run safety check --ignore-unpinned-requirements

# Pre-commit
pre-commit: ## Run pre-commit on all files
	$(UV) run pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	$(UV) run pre-commit autoupdate

# Comprehensive checks
check: ## Run all basic quality checks
	@echo "🔍 Running code quality checks..."
	@echo "📝 Checking format..."
	$(MAKE) format-check
	@echo "🔧 Running linting..."
	$(MAKE) lint
	@echo "🔍 Checking types..."
	$(MAKE) type-check
	@echo "🛡️  Checking security..."
	$(MAKE) security
	@echo "✅ All checks completed!"

all-checks: ## Run complete test suite including tests
	@echo "🚀 Running complete verification suite..."
	$(MAKE) check
	@echo "🧪 Running tests with coverage..."
	$(MAKE) test-cov
	@echo "🔒 Auditing dependencies..."
	$(MAKE) audit
	@echo "🎉 Complete suite executed successfully!"

ci: ## Run CI/CD checks
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security
	$(MAKE) test-cov
	$(MAKE) audit

# Development helpers
run: ## Run the main application
	$(UV) run python -m $(PACKAGE_NAME).main

build: ## Build the package
	$(UV) build

publish-test: ## Publish to TestPyPI
	$(UV) publish --repository testpypi

publish: ## Publish to PyPI
	$(UV) publish

# Environment
env-info: ## Show environment information
	@echo "Python version:"
	$(UV) run python --version
	@echo "UV version:"
	$(UV) --version
	@echo "Installed packages:"
	$(UV) pip list

update-deps: ## Update dependencies
	$(UV) lock --upgrade
