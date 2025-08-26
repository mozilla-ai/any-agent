# Contributing to mozilla.ai Any Agent

Thank you for your interest in contributing to this repository! This project supports the mozilla.ai goal of empowering developers to integrate AI capabilities into their projects using open-source tools and models.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

---

## **Guidelines for Contributions**

### Ground Rules

- Review issue discussion fully before starting work. Engage in the thread first when an issue is under discussion.
- PRs must build on agreed direction where ones exist. If there is no agreed direction, seek consensus from the core maintainers.
- PRs with "drive-by" unrelated changes or untested refactors will be closed.
- Untested or failing code is not eligible for review.
- PR description *must* follow the PR template and explain *what* changed, *why*, and *how to test*.
- Links to related issues are required.
- Duplicate PRs will be automatically closed.
- Only have 1-2 PRs open at a time. Any further PRs will be closed.

**Maintainers reserve the right to close issues and PRs that do not align with the library roadmap.**

### Code Clarity and Style
- **Readability first:** Code must be self-documenting—if it is not self-explanatory, it should include clear, concise comments where logic is non-obvious.
- **Consistent Style:** Follow existing codebase style (e.g., function naming, docstring format)
- **No dead/debug code:** Remove commented-out blocks, leftover print statements, unrelated refactors
- Failure modes must be documented and handled with robust exception handling.

For more details on writing self-documenting code, check out [this guide](https://swimm.io/learn/documentation-tools/tips-for-creating-self-documenting-code).

### Testing Requirements
- **Coverage:** All new functionality must include unit tests covering both happy paths and relevant edge cases.
- **Passing tests:** pre-commit must pass with all checks (see below on how to run).
- **No silent failures:** Tests should fail loudly on errors. No `assert True` placeholders.

### Scope and Size
- **One purpose per PR:** No kitchen-sink PRs mixing bugfixes, refactors, and features.
- **Small, reviewable chunks:** If your PR is too large to review in under 30 minutes, break it up into chunks.
    - Each chunk must be independently testable and reviewable
    - If you can't explain why it can't be split, expect an automatic request for refactoring.
- Pull requests that are **large** (>500 LOC changed) or span multiple subsystems will be closed with automatic requests for refactoring.
- If the PR is to implement a new feature, please first make a GitHub issue to suggest the feature and allow for discussion. We reserve the right to close feature implementations and request discussion via an issue.

## **How to Contribute**

### **Customize for your use-case or Extend It** 🔧
- Fork this repo and customize it for your own use-case or even extend its capabilities.
- We'd love to see what you've built!

### **Browse Existing Issues** 🔍
- Check the Issues page to see if there are any tasks you'd like to tackle.
- Look for issues labeled **`good first issue`** if you're new to the project—they're a great place to start.

### **Report Issues** 🐛
- Found a bug? Open a Bug Report by clicking on 'New Issue'
- Provide as much detail as possible, including the steps to reproduce the issue and Expected vs. actual behavior

### **Suggest Features** 🚀
- Have an idea for improving the project? Open a Feature Request by clicking on 'New Issue'
- Share why the feature is important and any alternative solutions you’ve considered.

### **Submit Pull Requests** 💻
- Fork the repository and create a new branch for your changes.
- Follow the [Guidelines for Contributions](#guidelines-for-contributions)
- Ensure your branch is up-to-date with the main branch before submitting the PR.
- Please follow the PR template, adding as much detail as possible, including how to test the changes

---

## **Guidelines for Contributions**

**Install**

We recommend to use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
source .venv/bin/activate
uv sync --dev --extra all
```

**Linting**

Ensure all the checks pass:

```bash
pre-commit run --all-files
```

**Testing**

Test changes locally to ensure functionality.

```bash
pytest -v tests
```

**Documentation**

Update docs for changes to functionality and maintain consistency with existing docs.

```bash
mkdocs serve
```
