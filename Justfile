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

# --- Main Recipes ---

# The default recipe, run when you just type `just`. It lists available commands.
default:
    @just --list

## ⚙️  setup: Create a uv virtual environment and install all dependencies.
setup:
    @echo ">>> Creating virtual environment in ./.venv..."
    @uv venv
    @echo "\n>>> Installing dependencies from requirements files..."
    @uv pip install -r requirements_docs.txt
    @echo "\n>>> Installing 'fluids' in editable mode..."
    @uv pip install -e .
    @echo "\n✅ Environment setup complete! You can now run other commands."

## 📚 docs: Build the Sphinx documentation.
docs:
    @echo ">>> Building Sphinx docs..."
    # This is correct because we want to run the `sphinx` installed inside the venv.
    @{{VENV_PYTHON}} -m sphinx -b html -d _build/doctrees docs _build/html -j auto
    @echo "✅ Docs built in _build/html"

## 🧪 test: Run the test suite with pytest.
test:
    @echo ">>> Running pytest..."
    @{{VENV_PYTEST}} -n auto

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

## 🧹 clean: Remove build artifacts and Python caches.
clean:
    @echo ">>> Cleaning up build artifacts and cache files..."
    @rm -rf _build .mypy_cache .pytest_cache dist *.egg-info
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Cleanup complete."

## 💣 nuke: Remove the virtual environment and all build artifacts.
nuke: clean
    @echo ">>> Removing virtual environment..."
    @rm -rf .venv
    @echo "✅ Project completely cleaned."