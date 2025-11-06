# Contributing to mozilla.ai Any Agent

Thank you for your interest in contributing to any-agent! 🎉

We're building a unified interface for working with multiple agent frameworks, supporting the mozilla.ai goal of empowering developers to integrate AI capabilities into their projects using open-source tools and models. We welcome contributions from developers of all experience levels—whether you're fixing a typo, adding a new framework integration, or improving our architecture, your help is appreciated.


## Before You Start

### Check for Duplicates

Before creating a new issue or starting work:
- [ ] Search [existing issues](https://github.com/mozilla-ai/any-agent/issues) for duplicates
- [ ] Check [open pull requests](https://github.com/mozilla-ai/any-agent/pulls) to see if someone is already working on it
- [ ] For bugs, verify it still exists in the `main` branch

### Discuss Major Changes First

For significant changes, please open an issue **before** starting work:

- New framework integrations
- API changes or new public methods
- Architectural changes
- Breaking changes
- New dependencies

**Use the `rfc` label** for design discussions. This ensures alignment with project goals and saves everyone time.

**Important**: PRs must build on agreed direction where one exists. If there is no agreed direction, seek consensus from the core maintainers before starting work.

### Read Our Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). We're committed to maintaining a welcoming, inclusive community.

## Development Setup

### Prerequisites

- **Python 3.11 or newer**
- **Git**
- **uv** (recommended) or your preferred package manager
- **API keys** for any LLM providers you want to test

### Quick Start

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) as your Python package and project manager.

```bash
# 1. Fork the repository on GitHub
# Click the "Fork" button at https://github.com/mozilla-ai/any-agent

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/any-agent.git
cd any-agent

# 3. Add upstream remote
git remote add upstream https://github.com/mozilla-ai/any-agent.git

# 4. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync --dev --extra all

# 5. Install pre-commit hooks
uv run pre-commit install

# 6. Verify your setup - ensure all checks pass
uv run pre-commit run --all-files

# 7. Run tests to confirm everything works
pytest -v tests
```

### Setting Up API Keys

Create a `.env` file in the project root (this file is gitignored):

```bash
# Add keys for providers you want to test
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
# Add others as needed
```

Alternatively, export environment variables:

```bash
export OPENAI_API_KEY="your_key_here"
```

**⚠️ Never commit API keys!** Always use environment variables or `.env` files.

## Making Changes

### 1. Create a Branch

Always work on a feature branch, never directly on `main`:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `framework/` - New framework integrations
- `refactor/` - Code improvements without behavior changes

### 2. Make Your Changes

Keep changes focused and well-documented.

### 3. Write Tests

**Every change needs tests!** This is non-negotiable.

#### Test Requirements

- **New features**: Add tests covering happy path and error cases
- **Bug fixes**: Add a test that reproduces the bug
- **Framework integrations**: Comprehensive test suite required
- **Coverage**: All new functionality must be thoroughly tested

Run tests locally:

```bash
# Run all tests
pytest -v tests

# Run specific test file
pytest -v tests/test_specific.py

# Run with coverage
pytest -v tests --cov
```

### 4. Ensure All Checks Pass

Before submitting, make sure all checks pass:

```bash
# Run pre-commit checks
pre-commit run --all-files

# Run tests
pytest -v tests
```

All checks must pass before your PR can be reviewed.

### 5. Update Documentation

Documentation is as important as code!

Update when you:
- Add a new feature
- Change existing behavior
- Add a new framework integration
- Fix a bug that affects usage

Documentation to update:
- **Docstrings** in code (required)
- **README.md** if changing core functionality
- **docs/** for any user-facing changes

Preview documentation locally:

```bash
mkdocs serve
```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "feat: add support for CrewAI framework"
git commit -m "fix: handle streaming responses correctly in tinyagent"
git commit -m "docs: update framework comparison table"

# Less helpful commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

## Submitting Your Contribution

### 1. Push to Your Fork

```bash
# Commit your changes
git add .
git commit -m "feat: add support for new framework"

# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create a Pull Request

1. Go to https://github.com/mozilla-ai/any-agent
2. Click "New Pull Request"
3. Click "compare across forks"
4. Select your fork and branch
5. Fill out the PR template completely
6. Click "Create Pull Request"

### 3. PR Description Requirements

Your PR description **must** include:

- **What changed**: Clear description of the changes
- **Why**: Explanation of the motivation and context
- **How to test**: Step-by-step instructions for testing your changes
- **Related issues**: Links to any related issues
- **Checklist completion**: All items in the PR template must be addressed

PRs without complete descriptions will be closed and asked to be resubmitted.

## Review Process

### What to Expect

1. **Initial Response**: Within **5 business days**
2. **Simple Fixes**: Usually merged within **1 week**
3. **Complex Features**: May take **2-3 weeks** for thorough review
4. **Framework Integrations**: Often require **2-3 review cycles**

### During Review

- Maintainers will provide constructive feedback
- Address comments with new commits (don't force push during review)
- Ask questions if feedback is unclear
- Be patient and respectful
- CI must pass before merge
- **All review comments must be addressed** before requesting re-review

### If Your PR Goes Stale

- No activity for **30+ days** may result in closure
- You can always reopen and continue later
- Let us know if you need help finishing
- We can find another contributor to complete it

## Your First Contribution

New to open source? Welcome! Here's how to get started:

### Step 1: Find an Issue

Look for issues labeled:
- `good-first-issue` - Perfect for newcomers
- `help-wanted` - Community contributions welcome
- `documentation` - Often accessible for beginners

### Step 2: Claim the Issue

Comment on the issue:
> "Hi! I'd like to work on this. Is it still available?"

We'll assign it to you and provide guidance.

### Step 3: Ask Questions Early

Don't spend days stuck! Ask questions:
- In the issue comments
- In GitHub Discussions
- Tag `@maintainers` if needed

We're here to help you succeed.

### Step 4: Start Small

Your first PR doesn't have to be ambitious:
- Fix a typo
- Improve documentation
- Add a test
- Fix a small bug

### Step 5: Learn and Grow

Every expert was once a beginner. We're here to help you grow as a contributor!

## Code of Conduct

This project follows Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

In brief:
- **Be respectful and inclusive**
- **Focus on constructive feedback**
- **Help create a welcoming environment**
- **Report concerns** to maintainers

See our full [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Questions?

- 💬 Open a [GitHub Discussion](https://github.com/mozilla-ai/any-agent/discussions)
- 🐛 Report a [Bug](https://github.com/mozilla-ai/any-agent/issues/new?template=bug_report.md)
- 💡 Request a [Feature](https://github.com/mozilla-ai/any-agent/issues/new?template=feature_request.md)

We're excited to have you as part of the any-agent community! 🚀

---

**License**: By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE) file).
