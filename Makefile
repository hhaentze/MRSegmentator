.PHONY: type pretty test_pretty smoke full compatibility test clean

type:
	mypy src --ignore-missing-imports --python-version=3.11 

pretty:
	isort --profile black src tests
	black --line-length 100 src tests

test_pretty:
	isort --check --profile black src tests
	black --line-length 100 --check src tests
	flake8-nb src tests

smoke:
	pytest tests/mrsegmentator/test_utils.py tests/mrsegmentator/test_smoke.py  tests/mrsegmentator/test_weights.py -v

full:
	pytest tests/mrsegmentator/ -v

compatibility:
	python tests/compatibility/run_matrix.py

test: smoke
	mypy src --ignore-missing-imports --python-version=3.11
	
clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.monai-cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.mypy_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('build')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('dist')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('htmlcov')]"
	@echo "Clean complete."