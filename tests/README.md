## Running tests

```bash
git clone https://github.com/speediedan/finetuning-scheduler
cd finetuning-scheduler

# install dev deps
python -m pip install ".[all]"

# run tests
pytest -v
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have at least 2 GPUs to run distributed tests.

Note that this setup will not run tests that require specific packages installed.

## Running Coverage

### To generate full-coverage (requires minimum of 2 GPUS)

```bash
cd finetuning-scheduler
python -m coverage erase && \
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v && \
(./tests/standalone_tests.sh -k test_f --no-header 2>&1 > /tmp/standalone.out) > /dev/null && \
egrep '(Running|passed|failed|error)' /tmp/standalone.out && \
python -m coverage report -m
```

### To generate cpu-only coverage:

```bash
cd finetuning-scheduler
python -m coverage run --source src/finetuning_scheduler -m pytest src/finetuning_scheduler tests -v
```
