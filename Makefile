type:
	mypy src --strict --ignore-missing-imports --python-version=3.11

pretty:
	isort --profile black src
	black --line-length 100 src

test_pretty:
	isort --check --profile black src
	black --line-length 100 --check src

test:
	flake8-nb src
	mypy src --strict --ignore-missing-imports --python-version=3.11

clean:
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.monai-cache')]"
