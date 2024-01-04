pretty:
	isort .
	black --line-length 100 .

test_pretty:
	isort --check --profile black .
	black --line-length 100 --check .

test:
	flake8-nb

clean:
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"
	python3 -Bc "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.monai-cache')]"
