.PHONY: test clean docs

# initially based on https://bit.ly/3rFergQ
# assume you have installed needed packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf $(shell find . -name "lightning_log")
	rm -rf $(shell find . -name "lightning_logs")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

test: clean
	pip install -r requirements/devel.txt

	# run tests with coverage
	python -m coverage run --source finetuning_scheduler -m pytest tests fts_examples -v
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build
