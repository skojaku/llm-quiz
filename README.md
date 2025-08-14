# LLM Quiz Challenge

Learning to create good questions is a great learning exercise! Creating good questions requires a deep understanding of the materials and attention to details, edge cases, and subtle distinctions. This led me to think: why not ask students to create questions, instead of me generating them?

This repository provides a simple command line tool that lets students create questions that stump large language models. Large language models are pretty good at answering questions about surface-level knowledge, but often fail short for questions that require nuanced understanding. Through this exercise, I hope students develop a deeper understanding and attention to details, along with the limitations of large language models they might use daily in their work!

## How It Works


![Demo](./demo.gif)

Students create quiz questions in a TOML file, then the tool uses AI to try answering them. If the AI fails, the student wins!

In my lecture, I integrate this tool with GitHub Actions to automatically generate pass/fail markers for students when they submit their quiz. You can find [an example here](https://github.com/sk-classroom/advnetsci-robustness).

## Quick Start

1. **Create a quiz file** (`quiz.toml`):
```toml
[[questions]]
question = "When is the global clustering not a good representation of the network?"
answer = "When the network is degree heterogeneous. Hubs can create many triangles, not representing typical nodes."

[[questions]]
question = "When is the average path length not a good representation of the network?"
answer = "When the network is degree heterogeneous. Hubs can substantially reduce average path lengths."
```

2. Get an API key from [OpenRouter](https://openrouter.ai/api-key). Alternatively, you can use [Ollama](https://ollama.ai/) to run the tool locally.

3. **Run the tool**:
```bash
uv run python -m llm_quiz.cli --quiz-file quiz.toml --api-key your-api-key
```

(Install `uv` first: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

4. **See results**:
```
🎉 RESULT: PASS - You successfully stumped the AI!
📊 Summary: 2/2 questions stumped the AI
Success Rate: 100.0%
```

## Installation

```bash
# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv install

# Or with pip
uv pip install -r llm_quiz/requirements.txt
```

## Configuration

Create a `config.toml` file:
```toml
[api]
base_url = "https://openrouter.ai/api/v1"

[models]
quiz_model = "openrouter/google/gemma-3-4b-it"
evaluator_model = "openrouter/google/gemini-2.5-flash-lite"
```

## Command Options

```bash
# Basic usage
uv run python -m llm_quiz.cli --quiz-file quiz.toml --api-key sk-xxx

# Save results
uv run python -m llm_quiz.cli --quiz-file quiz.toml --api-key sk-xxx --output results.json

# Use different models
uv run python -m llm_quiz.cli --quiz-file quiz.toml --api-key sk-xxx --quiz-model gpt-4o-mini

# Local Ollama
uv run python -m llm_quiz.cli --quiz-file quiz.toml --base-url http://localhost:11434/v1 --api-key dummy --quiz-model llama2
```
