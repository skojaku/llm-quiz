# GitHub Actions Setup

This repository includes a CI workflow that tests the LLM Quiz Challenge installation and functionality.

## Setting up the API Key Secret

To enable full integration testing with real API calls:

1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Name: `API_KEY`
5. Value: Your OpenRouter API key (or other compatible API key)

The workflow will automatically use this secret when available. If no API key is provided, the workflow will still run basic tests and validation.

## Workflow Features

The CI workflow includes:

- **Multi-version Python testing** (3.9, 3.10, 3.11, 3.12)
- **Dependency installation** using uv package manager
- **Basic functionality tests** without API requirements
- **Code quality checks** (black, isort, flake8)
- **CLI installation verification**
- **Integration testing** with real API (when secret is available)

## Manual Testing

To test the workflow locally:

```bash
# Install dependencies
uv sync --dev

# Run basic tests
uv run python -m llm_quiz.test_dspy

# Run pytest
uv run pytest llm_quiz/ -v

# Check code quality
uv run black --check llm_quiz/
uv run isort --check-only llm_quiz/
uv run flake8 llm_quiz/
```