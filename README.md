# Being Robust (in High Dimensions) Can Be Practical

![actions](https://github.com/GeislerSimon/being-robust-can-be-practical/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/GeislerSimon/being-robust-can-be-practical/branch/master/graph/badge.svg)](https://codecov.io/gh/GeislerSimon/being-robust-can-be-practical)

Minimal implementation of 
> Ilias Diakonikolas, Gautam Kamath, Daniel M. Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Being robust 
(in high dimensions) can be practical. 34th International Conference on Machine Learning, ICML 2017, 3:1659–1689, 2017.

I closely followed the Matlab implementation. So I rather refer to the original code than the paper.

## Installation

For the required dependencies please use:
```bash
pip install -r requirements.txt
```

With
```bash
pip install -r requirements-dev.txt
```
you install the development dependencies, e.g. for running unit tests.

Alternatively, you can use [poetry](https://python-poetry.org/docs/) for installing the dependencies:

```bash
poetry install
```

## Getting started

Examples and description of the parameters are given in the docstrings.

## Contributing

Feel free to open pull requests in case you would like to contribute. Please follow the 
[angular commit message style](https://github.com/angular/angular/blob/master/CONTRIBUTING.md).
