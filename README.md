# LLM Quiz Challenge

Writing great questions is one of the best ways to deepen your understanding of a subject.
As an instructor, I often found myself reviewing course materials to ensure my quiz questions went beyond surface-level facts and required nuanced understanding.
This inspired me to turn the tables! What if students created their own challenging questions as a way to learn more effectively?

This repository provides a simple command line tool that lets students create questions that stump large language models. Large language models are pretty good at answering questions about surface-level knowledge, but often fail short for questions that require nuanced understanding. Through this exercise, I hope students develop a deeper understanding and attention to details, along with the limitations of large language models they might use daily in their work.

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

Create a `config.toml` file to customize the tool's behavior:

```toml
[api]
base_url = "https://openrouter.ai/api/v1"

[models]
quiz_model = "openrouter/google/gemma-3-4b-it"
evaluator_model = "openrouter/google/gemini-2.5-flash-lite"

[context]
urls = [
    "https://raw.githubusercontent.com/course/repo/main/docs/lecture-notes.qmd",
    "https://raw.githubusercontent.com/course/repo/main/slides/week1.md"
]

[output]
verbose = false
```

### URL Grounding

The `[context.urls]` section is particularly useful for course integration. By providing URLs to your course materials (lecture notes, slides, readings), the tool can:

- **Validate question relevance**: Ensure student questions are actually about your course content
- **Provide better context**: Help the AI understand the specific terminology and concepts you're teaching
- **Improve evaluation accuracy**: Give the evaluator model the same context students have access to

Supported URL formats:
- Raw GitHub files (recommended): `https://raw.githubusercontent.com/user/repo/main/file.md`
- Web pages: `https://course-website.com/readings/`

The tool will fetch and process these materials to create a knowledge base for validating and evaluating student questions.
PDFs are not currently supported but might be in the future.

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
