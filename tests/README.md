## Running tests

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# install dev deps
# PACKAGE_NAME variable currently required to specify pytorch-lightning dev package dep (as of lightning 1.8.0)
export PACKAGE_NAME=pytorch
python -m pip install ".[all]"

# run tests
pytest -v
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have at least 2 GPUs to run distributed tests.

Note that this setup will not run tests that require specific packages installed
such as Horovod, FairScale, NVIDIA/apex, NVIDIA/DALI, etc.
You can rely on our CI to make sure all these tests pass.

## Running Coverage

Make sure to run coverage on a GPU machine with at least 2 GPUs

```bash
cd finetuning-scheduler

# generate coverage (coverage is also installed as part of dev dependencies under requirements/devel.txt)
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v

# print coverage stats
coverage report -m

# exporting results
coverage xml
```
