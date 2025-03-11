# autogen-selector


## Setup

1. Clone the repository
1. `cd autogen-selector` (root directory of this git repository)
1. `uv sync`
1. `source ./.venv/bin/activate` or `.venv/bin/activate` (activate the virtual environment)
1. `pre-commit install`
1. `cp .env.sample .env` (fill in the values)
1. code . (open the project in vscode)
1. install the recommended extensions (cmd + shift + p -> `Extensions: Show Recommended Extensions`)

## Samples
```sh
python -m autogen_selector.main
```

## Unit Test Coverage

```sh
python -m pytest -p no:warnings --cov-report term-missing --cov=autogen_selector tests
```