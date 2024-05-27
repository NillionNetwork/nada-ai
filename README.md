# Nada AI

![GitHub License](https://img.shields.io/github/license/NillionNetwork/nada-ai?style=for-the-badge)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/NillionNetwork/nada-ai/test?style=for-the-badge)

Nada AI is a Python library designed for performing ML/AI on top of Nada DSL and the Nillion Network.

It provides an intuitive ML interface and frictionless integration with existing ML frameworks such as PyTorch and Sci-kit learn.

## Features

- **Exporting model state**: Integrates with models from existing ML frameworks and provides an easy way to export them to the Nillion network - to be used in Nada programs.
- **AI Modules**: A PyTorch-esque interface to create arbitrary ML models in Nada by stacking pre-built common ML components - with the possibility of easily creating custom components.
- **Importing model state**: Easily import an exported model state that lives in the Nillion network to be used in a Nada program.

## Installation
### Using pip

```bash
pip install nada-ai
```

### From Sources
You can install the nada-algebra library using Poetry:

```bash
git clone https://github.com/NillionNetwork/nada-ai.git
pip3 install poetry
poetry install nada-ai
```

## License

This project is licensed under the Apache2 License. See the LICENSE file for details.
