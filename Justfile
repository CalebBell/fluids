# Use a strict shell for more predictable recipe execution.
# The "-c" is crucial: it tells bash to treat the recipe lines as commands.
set shell := ["bash", "-euo", "pipefail", "-c"]

# --- Variables ---
# Define paths to executables within the virtual environment for clarity.
# This ensures we always use the tools installed in our project's venv.
VENV_PYTHON := ".venv/bin/python"
VENV_PYTEST := ".venv/bin/pytest"
VENV_MYPY   := ".venv/bin/mypy"
VENV_RUFF   := ".venv/bin/ruff"
VENV_PIP_AUDIT := ".venv/bin/pip-audit"
VENV_BANDIT := ".venv/bin/bandit"

# --- Main Recipes ---

# The default recipe, run when you just type `just`. It lists available commands.
default:
    @just --list

## ⚙️  install: Create a uv virtual environment and install all dependencies.
install:
    @echo ">>> Creating virtual environment in ./.venv..."
    @uv venv
    @echo "\n>>> Installing 'fluids' in editable mode with dev dependencies..."
    @uv pip install -e .[dev]
    @echo "\n>>> Installing prek hooks..."
    @prek install
    @echo "\n✅ Environment setup complete! You can now run other commands."

## 📚 docs: Build the Sphinx documentation.
docs:
    @echo ">>> Building Sphinx docs..."
    @{{VENV_PYTHON}} -m sphinx -b html -d _build/doctrees docs _build/html -j auto
    @echo "✅ Docs built in _build/html"

## 🧪 test: Run the test suite with pytest.
test *ARGS:
    @echo ">>> Running pytest..."
    @{{VENV_PYTEST}} -n auto {{ARGS}}

## 📊 test-cov: Run tests with coverage report.
test-cov:
    @echo ">>> Running pytest with coverage..."
    @{{VENV_PYTEST}} -n auto --cov=fluids --cov-report=html --cov-report=term
    @echo "✅ Coverage report generated in htmlcov/"

## 🧐 typecheck: Check static types with mypy.
typecheck:
    @echo ">>> Running mypy..."
    @{{VENV_MYPY}} .

## ✨ lint: Check for code style issues and errors with Ruff.
lint:
    @echo ">>> Running Ruff..."
    @{{VENV_RUFF}} check .

## 🏁 check: Run all checks (linting and type checking).
check: lint typecheck

## 🔒 security: Run security scans with pip-audit and bandit.
security:
    @echo ">>> Running pip-audit..."
    @{{VENV_PIP_AUDIT}} -r requirements.txt
    @echo ">>> Running bandit..."
    @{{VENV_BANDIT}} -r fluids -ll
    @echo "✅ Security scans complete."

## 🪝 precommit: Run pre-commit hooks on all files.
precommit:
    @echo ">>> Running pre-commit hooks..."
    @prek run --all-files

## 🔌 hooks-install: Install prek hooks.
hooks-install:
    @echo ">>> Installing prek hooks..."
    @prek install
    @echo "✅ Hooks installed."

## 🗑️  hooks-remove: Remove prek hooks.
hooks-remove:
    @echo ">>> Removing prek hooks..."
    @prek uninstall
    @echo "✅ Hooks removed."

# asv is broken
# ## ⚡ bench: Run performance benchmarks.
# bench:
#     @echo ">>> Running benchmarks..."
#     @asv run

## 📦 build: Build wheel and source distributions.
build:
    @echo ">>> Building distributions..."
    @{{VENV_PYTHON}} -m build
    @echo "✅ Distributions built in dist/"

## 🔍 check-dist: Check built distributions with twine.
check-dist:
    @echo ">>> Checking distributions with twine..."
    @.venv/bin/twine check dist/*
    @echo "✅ Distributions are valid."

## 🚀 ci: Run all CI checks (lint, typecheck, test).
ci: lint typecheck test
    @echo "✅ All CI checks passed!"

## 🧹 clean: Remove build artifacts and Python caches.
clean:
    @echo ">>> Cleaning up build artifacts and cache files..."
    @rm -rf _build .mypy_cache .pytest_cache dist *.egg-info htmlcov prof
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Cleanup complete."

## 💣 nuke: Remove the virtual environment and all build artifacts.
nuke: clean
    @echo ">>> Removing virtual environment..."
    @rm -rf .venv
    @echo "✅ Project completely cleaned."