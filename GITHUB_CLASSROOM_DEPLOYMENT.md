# GitHub Classroom Deployment Instructions

This document provides step-by-step instructions for deploying the LLM Quiz Challenge tool as a GitHub Action with GitHub Classroom for automated grading.

## Overview

The GitHub Classroom integration automatically grades student quiz submissions by:
1. Running the LLM quiz challenge on student-submitted `quiz.toml` files
2. Evaluating whether students successfully stumped the AI models
3. Providing pass/fail results with detailed feedback
4. Generating scores for the GitHub Classroom gradebook

## Prerequisites

- GitHub Classroom assignment setup
- OpenRouter API account and key (or alternative LLM API)
- Repository with the LLM Quiz Challenge tool

## Setup Instructions

### 1. GitHub Classroom Assignment Configuration

1. Create a new assignment in GitHub Classroom
2. Use this repository as the starter code
3. Enable automatic grading in the assignment settings
4. Set the maximum points (recommended: 10 points)

### 2. Repository Secrets Configuration

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (starts with `sk-or-v1-`) | Yes |

**To add secrets:**
1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with the exact name shown above

### 3. Workflow File

The workflow file is already included in this repository at `.github/workflows/classroom.yml`. No changes are needed for basic deployment.

### 4. Configuration Customization

#### A. Update Course-Specific Context (Required)

Edit `config.toml` to include your course materials:

```toml
[context]
# Context materials - URLs to fetch for providing context to the quiz model
urls = [
    "https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-COURSE-REPO/main/docs/lecture1.md",
    "https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-COURSE-REPO/main/slides/week1.md",
    # Add more URLs as needed
]
```

**Important:** Replace the example URLs with actual links to your course materials. These should be:
- Raw GitHub file URLs (not regular GitHub page URLs)
- Publicly accessible (no authentication required)
- Relevant to your course content

#### B. Adjust Models (Optional)

If you want to use different AI models, edit the `[models]` section in `config.toml`:

```toml
[models]
quiz_model = "openrouter/google/gemma-3-4b-it"        # Model for answering questions
evaluator_model = "openrouter/google/gemini-2.5-flash-lite"        # Model for evaluation
```

Available options include:
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4o` - Higher quality but more expensive
- `openrouter/google/gemma-3-4b-it` - Good balance (default)
- `openrouter/google/gemini-2.5-flash-lite` - Fast evaluation

#### C. Adjust Scoring (Optional)

To change the maximum points, edit `.github/workflows/classroom.yml`:

```yaml
- name: Quiz Challenge
  with:
    max-score: 10  # Change this number
```

### 5. Student Submission Format

Students should submit their quiz questions in a `quiz.toml` file in the repository root:

```toml
[[questions]]
question = "Your challenging question here"
answer = "The correct answer that requires deep understanding"

[[questions]]
question = "Another question that tests subtle distinctions"
answer = "Another nuanced answer that shows comprehension"
```

## File Structure

After setup, your repository should have this structure:

```
your-repo/
├── .github/
│   └── workflows/
│       └── classroom.yml          # GitHub Actions workflow (DO NOT MODIFY)
├── llm_quiz/                      # Quiz tool source code (DO NOT MODIFY)
├── config.toml                    # Configuration file (CUSTOMIZE THIS)
├── quiz.toml                      # Student's quiz file (students create this)
├── requirements.txt               # Python dependencies (DO NOT MODIFY)
├── pyproject.toml                # Python project config (DO NOT MODIFY)
└── README.md                     # Instructions for students
```

## Customization Variables Summary

### Must Change:
1. **Context URLs in config.toml** - Replace with your course material URLs
2. **GitHub Secrets** - Add your API key

### May Change:
1. **Models in config.toml** - Adjust AI models based on your needs
2. **Max score in classroom.yml** - Change point values
3. **Timeout in classroom.yml** - Adjust if quizzes need more time (default: 10 minutes)

### Advanced Customization:
1. **API endpoint** - If using a different LLM provider, update `base_url` in config.toml
2. **Secret name** - If you prefer a different secret name, update both the workflow and add the corresponding secret

## Grading Logic

The autograding system works as follows:

- **Pass (Full Points)**: Student successfully stumps the AI on all valid questions (100% win rate)
- **Fail (0 Points)**: Student fails to stump the AI on one or more questions, or has invalid questions

Students receive detailed feedback including:
- Which questions stumped the AI (✅ wins)
- Which questions the AI answered correctly (❌ losses)  
- Specific suggestions for improving questions that didn't work
- Overall success rate

## Troubleshooting

### Common Issues:

1. **"OPENROUTER_API_KEY secret not found"**
   - Solution: Add the API key secret in repository settings

2. **Workflow times out**
   - Solution: Increase timeout value in `classroom.yml` (max recommended: 15)

3. **API quota exceeded**
   - Solution: Check your OpenRouter usage and billing

4. **Context URLs failing**
   - Solution: Ensure URLs are raw GitHub links and publicly accessible

5. **Students get 0 points despite good questions**
   - Solution: Check that context URLs are relevant to the questions being asked

### Debug Mode:

To enable verbose logging for troubleshooting, temporarily edit `config.toml`:

```toml
[output]
verbose = true
```

## Cost Estimation

Typical costs per student submission:
- With default models: ~$0.01-0.05 per quiz
- With GPT-4: ~$0.10-0.25 per quiz

Costs depend on:
- Number of questions per quiz
- Length of context materials
- Models chosen
- Question complexity

For a class of 50 students with 3 quiz submissions each, expect approximately $1.50-37.50 in API costs for the semester with default models.