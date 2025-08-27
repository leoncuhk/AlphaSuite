# Contributing to AlphaSuite

First off, thank you for considering contributing to AlphaSuite! It's people like you that make open-source projects such a great place to learn, inspire, and create. Any contribution, no matter how small, is greatly appreciated.

This document provides guidelines for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by the [AlphaSuite Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

There are many ways to contribute, from writing code and documentation to submitting bug reports and feature requests.

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/rsandx/AlphaSuite/issues). If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on GitHub. This allows for a discussion of your idea before you start working on it.

### Submitting Pull Requests

Ready to contribute some code? Great! Here’s how to set up AlphaSuite for local development and submit your changes.

## Development Setup

1.  Fork the `AlphaSuite` repository on GitHub.
2.  Clone your fork locally:
    ```bash
    git clone https://github.com/your_name_here/AlphaSuite.git
    ```
3.  Follow the installation and setup instructions in the main README.md to create your virtual environment and install dependencies.
4.  Create a new branch for your local development:
    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```
    Now you can make your changes locally.

## Pull Request Process

1.  When you're done making changes, ensure your code follows the project's style guidelines.
2.  Commit your changes using a descriptive commit message.
    ```bash
    git add .
    git commit -m "feat: Add a new awesome feature"
    ```
3.  Push your branch to your fork on GitHub:
    ```bash
    git push origin name-of-your-bugfix-or-feature
    ```
4.  Open a pull request to the `main` branch of the `rsandx/AlphaSuite` repository.
5.  Provide a clear title and description for your pull request, explaining the changes you've made and why.

## Contributing a New Trading Strategy

One of the best ways to contribute to AlphaSuite is by adding a new trading strategy. The platform is designed to make this as easy as possible.

The core idea is to create a self-contained Python file in the `strategies/` directory that defines a class inheriting from `BaseStrategy`. The system will automatically discover and load it.

For a detailed, step-by-step guide on how to structure your strategy file and implement the required methods, please refer to the **"Adding a New Trading Strategy"** section in the main README.md. We've provided a comprehensive breakdown of each required method with examples.

We are excited to see what new strategies you come up with!