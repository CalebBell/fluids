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

## 🧊 test-cxfreeze: Test cx_Freeze compatibility (build executable and run it).
test-cxfreeze:
    @echo ">>> Creating temporary virtual environment..."
    @uv venv .venv-cxfreeze
    @echo "\n>>> Installing project and cx_Freeze in temporary environment..."
    @uv pip install --python .venv-cxfreeze/bin/python -e .[test]
    @uv pip install --python .venv-cxfreeze/bin/python cx_Freeze
    @echo "\n>>> Building cx_Freeze executable..."
    @cd dev && ../.venv-cxfreeze/bin/python cx_freeze_basic_standalone_check_builder.py build && cd ..
    @echo "\n>>> Testing executable..."
    @./dev/build/exe.*/basic_standalone_fluids_check
    @echo "\n>>> Cleaning up temporary environment and build artifacts..."
    @rm -rf .venv-cxfreeze dev/build
    @echo "✅ cx_Freeze test complete and cleaned up!"

## 🔥 test-nuitka: Test Nuitka compatibility (compile module and import it).
test-nuitka:
    @echo ">>> Creating temporary virtual environment..."
    @uv venv .venv-nuitka
    @echo "\n>>> Installing project and Nuitka in temporary environment..."
    @uv pip install --python .venv-nuitka/bin/python -e .[test,numba]
    @uv pip install --python .venv-nuitka/bin/python nuitka
    @echo "\n>>> Creating temporary test directory..."
    @mkdir -p .nuitka-test
    @cp -r fluids .nuitka-test/
    @echo "\n>>> Building Nuitka module in temporary directory..."
    @cd .nuitka-test && ../.venv-nuitka/bin/python -m nuitka --module fluids --include-package=fluids
    @echo "\n>>> Removing original fluids folder from test directory..."
    @rm -rf .nuitka-test/fluids/fluids
    @echo "\n>>> Testing compiled module can be imported..."
    @cd .nuitka-test && ../.venv-nuitka/bin/python -c "import fluids; print('Version:', fluids.__version__)"
    @echo "\n>>> Cleaning up temporary environment and build artifacts..."
    @rm -rf .venv-nuitka .nuitka-test
    @echo "✅ Nuitka test complete and cleaned up!"

## 📦 test-pyinstaller: Test PyInstaller compatibility (build executable and run it).
test-pyinstaller:
    @echo ">>> Creating temporary virtual environment..."
    @uv venv .venv-pyinstaller
    @echo "\n>>> Installing project and PyInstaller in temporary environment..."
    @uv pip install --python .venv-pyinstaller/bin/python .[test]
    @uv pip install --python .venv-pyinstaller/bin/python pyinstaller
    @echo "\n>>> Creating temporary test directory..."
    @mkdir -p .pyinstaller-test
    @cp -r dev .pyinstaller-test/
    @echo "\n>>> Building PyInstaller executable..."
    @cd .pyinstaller-test/dev && ../../.venv-pyinstaller/bin/pyinstaller --onefile --name basic_standalone_fluids_check basic_standalone_fluids_check.py
    @echo "\n>>> Testing executable..."
    @./.pyinstaller-test/dev/dist/basic_standalone_fluids_check
    @echo "\n>>> Cleaning up temporary environment and build artifacts..."
    @rm -rf .venv-pyinstaller .pyinstaller-test
    @echo "✅ PyInstaller test complete and cleaned up!"

## 🧹 clean: Remove build artifacts and Python caches.
clean:
    @echo ">>> Cleaning up build artifacts and cache files..."
    @rm -rf _build .mypy_cache .pytest_cache dist *.egg-info htmlcov prof dev/build .venv-cxfreeze .venv-nuitka .nuitka-test .venv-pyinstaller .pyinstaller-test
    @rm -f fluids.*.so fluids.*.pyd
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Cleanup complete."

## 💣 nuke: Remove the virtual environment and all build artifacts.
nuke: clean
    @echo ">>> Removing virtual environment..."
    @rm -rf .venv
    @echo "✅ Project completely cleaned."