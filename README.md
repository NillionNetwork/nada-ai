# Nada AI

![GitHub License](https://img.shields.io/github/license/NillionNetwork/nada-ai?style=for-the-badge&logo=apache&logoColor=white&color=%23D22128&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-ai%2Fblob%2Fmain%2FLICENSE&link=https%3A%2F%2Fgithub.com%2FNillionNetwork%2Fnada-ai%2Fblob%2Fmain%2FLICENSE)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/NillionNetwork/nada-ai/test.yml?style=for-the-badge&logo=python&logoColor=white&link=https://github.com/NillionNetwork/nada-ai/actions/workflows/test.yml&link=https://github.com/NillionNetwork/nada-ai/actions/workflows/test.yml)
![GitHub Release](https://img.shields.io/github/v/release/NillionNetwork/nada-ai?sort=date&display_name=release&style=for-the-badge&logo=dependabot&label=LATEST%20RELEASE&color=0000FE&link=https%3A%2F%2Fpypi.org%2Fproject%2Fnada-ai&link=https%3A%2F%2Fpypi.org%2Fproject%2Fnada-ai)

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
You can install the nada-ai library using Poetry:

```bash
git clone https://github.com/NillionNetwork/nada-ai.git
pip install ./nada-ai
```
### Advanced Options

In certain cases, you may want to install Nada AI with different development libraries. The options may be installed as follows:
```
pip install ./nada-ai[examples] # To include the libraries required for the examples
pip install ./nada-ai[linter] # To include automatic linting tools for development
```

In a normal scenario, these libraries won't be installed by Nada AI.
## License

This project is licensed under the Apache2 License. See the LICENSE file for details.
