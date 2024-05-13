type:
	mypy src --ignore-missing-imports --python-version=3.11 

pretty:
	isort --profile black src tests
	black --line-length 100 src tests

test_pretty:
	isort --check --profile black src tests
	black --line-length 100 --check src tests
	flake8-nb src tests

test:
	mypy src --ignore-missing-imports --python-version=3.11 
	python -m unittest tests/mrsegmentator/test_*.py -v

clean:
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.monai-cache')]"
