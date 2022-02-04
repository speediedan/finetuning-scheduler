# Contributing

Welcome to the community! Finetuning Scheduler extends the most advanced DL research platform on the planet (PyTorch Lightning) and strives to support the latest, best practices and integrations that the amazing PyTorch team and other research organizations roll out!

As Finetuning Scheduler is an extension of PyTorch Lightning, the remainder of the contribution guidelines conform to (and many are drawn from) the PyTorch Lightning [contribution documentation](https://pytorch-lightning.readthedocs.io/en/latest/generated/CONTRIBUTING.html).

A giant thank you to the [PyTorch Lightning team](https://pytorch-lightning.readthedocs.io/en/latest/governance.html#leads) for their tireless effort building the immensely useful PyTorch Lightning project and their thoughtful feedback on and review of this extension.

## Main Core Value: One less thing to remember

Simplify the API as much as possible from the user perspective.
Any additions or improvements should minimize the things the user needs to remember.

## Design Principles

We encourage all sorts of contributions you're interested in adding! When coding for Finetuning Scheduler, please follow these principles.

### No PyTorch Interference

We don't want to add any abstractions on top of pure PyTorch.
This gives researchers all the control they need without having to learn yet another framework.

### Simple Internal Code

It's useful for users to look at the code and understand very quickly what's happening.
Many users won't be engineers. Thus we need to value clear, simple code over condensed ninja moves.
While that's super cool, this isn't the project for that :)

### Simple External API

What makes sense to you may not make sense to others. When creating an issue with an API change suggestion, please validate that it makes sense for others.
Treat code changes the way you treat a startup: validate that it's a needed feature, then add if it makes sense for many people.

### Backward-compatible API

We all hate updating our deep learning packages because we don't want to refactor a bunch of stuff. With the Finetuning Scheduler, we make sure every change we make which could break an API is backward compatible with good deprecation warnings.

**You shouldn't be afraid to upgrade the Finetuning Scheduler :)**

### Gain User Trust

As a researcher, you can't have any part of your code going wrong. So, make thorough tests to ensure that every implementation of a new trick or subtle change is correct.

______________________________________________________________________

## Contribution Types

We are always open to contributions of new features or bug fixes.

A lot of good work has already been done in project mechanics (requirements.txt, setup.py, pep8, badges, ci, etc...) so we're in a good state there thanks to all the early contributors (even pre-beta release)!

### Bug Fixes:

1. If you find a bug please submit a GitHub issue.

   - Make sure the title explains the issue.
   - Describe your setup, what you are trying to do, expected vs. actual behaviour. Please add configs and code samples.
   - Add details on how to reproduce the issue - a minimal test case is always best, colab is also great.
     Note, that the sample code shall be minimal and if needed with publicly available data.

1. Try to fix it or recommend a solution. We highly recommend to use test-driven approach:

   - Convert your minimal code example to a unit/integration test with assert on expected results.
   - Start by debugging the issue... You can run just this particular test in your IDE and draft a fix.
   - Verify that your test case fails on the main branch and only passes with the fix applied.

1. Submit a PR!

_**Note**, even if you do not find the solution, sending a PR with a test covering the issue is a valid contribution, and we can help you or finish it with you :\]_

### New Features:

1. Submit a GitHub issue - describe what is the motivation of such feature (adding the use case, or an example is helpful).

1. Determine the feature scope with us.

1. Submit a PR! We recommend test driven approach to adding new features as well:

   - Write a test for the functionality you want to add.
   - Write the functional code until the test passes.

1. Add/update the relevant tests!

### Test cases:

Want to keep Finetuning Scheduler healthy? Love seeing those green tests? So do we! How to we keep it that way? We write tests! We value tests contribution even more than new features.

______________________________________________________________________

## Guidelines

### Developments scripts

To build the documentation locally, simply execute the following commands from project root (only for Unix):

- `make clean` cleans repo from temp/generated files
- `make docs` builds documentation under _docs/build/html_
- `make test` runs all project's tests with coverage

### Original code

All added or edited code shall be the own original work of the particular contributor.
If you use some third-party implementation, all such blocks/functions/modules shall be properly referred and if possible also agreed by code's author. For example - `This code is inspired from http://...`.

### Coding Style

1. Use f-strings for output formation
1. You can use [pre-commit](https://pre-commit.com/) to make sure your code style is correct.

### Documentation

We are using Sphinx with Napoleon extension.
Moreover, we set Google style to follow with type convention.

- [Napoleon formatting with Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [ReStructured Text (reST)](https://docs.pylonsproject.org/projects/docs-style-guide/)
- [Paragraph-level markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#paragraphs)

See following short example of a sample function taking one position string and optional

```python
from typing import Optional


def my_func(param_a: int, param_b: Optional[float] = None) -> str:
    """Sample function.

    Args:
        param_a: first parameter
        param_b: second parameter

    Return:
        sum of both numbers

    Example::

        Sample doctest example...
        >>> my_func(1, 2)
        3

    Note:
        If you want to add something.
    """
    p = param_b if param_b else 0
    return str(param_a + p)
```

When updating the docs make sure to build them first locally and visually inspect the html files (in the browser) for
formatting errors. In certain cases, a missing blank line or a wrong indent can lead to a broken layout.
Run these commands

```bash
pip install -r requirements/docs.txt
make clean
cd docs
make html
```

and open `docs/build/html/index.html` in your browser.

Notes:

- You need to have LaTeX installed for rendering math equations. You can for example install TeXLive by doing one of the following:
  - on Ubuntu (Linux) run `apt-get install texlive` or otherwise follow the instructions on the TeXLive website
  - use the [RTD docker image](https://hub.docker.com/r/readthedocs/build)
- with PL used class meta you need to use python 3.7 or higher

### Testing

**Local:** Testing your work locally will help you speed up the process since it allows you to focus on particular (failing) test-cases.
To setup a local development environment, install both local and test dependencies:

```bash
python -m pip install ".[all]"
python -m pip install pre-commit
pre-commit install
```

Note: if your computer does not have multi-GPU nor TPU these tests are skipped.

**GitHub Actions:** For convenience, you can also use your own GHActions building which will be triggered with each commit.
This is useful if you do not test against all required dependency versions.

You can then run:

```bash
python -m pytest finetuning_scheduler tests fts_examples -v
```

### Pull Request

We welcome any useful contribution! For your convenience here's a recommended workflow:

1. Think about what you want to do - fix a bug, repair docs, etc. If you want to implement a new feature or enhance an existing one.

   - Start by opening a GitHub issue to explain the feature and the motivation.
     In the case of features, ask yourself first - Is this NECESSARY for Finetuning Scheduler? There are some PRs that are just
     purely about adding engineering complexity which has no place in Finetuning Scheduler.
   - Core contributors will take a look (it might take some time - we are often overloaded with issues!) and discuss it.
   - Once an agreement was reached - start coding.

1. Start your work locally.

   - Create a branch and prepare your changes.
   - Tip: do not work on your main branch directly, it may become complicated when you need to rebase.
   - Tip: give your PR a good name! It will be useful later when you may work on multiple tasks/PRs.

1. Test your code!

   - It is always good practice to start coding by creating a test case, verifying it breaks with current behavior, and passes with your new changes.
   - Make sure your new tests cover all different edge cases.
   - Make sure all exceptions raised are tested.
   - Make sure all warnings raised are tested.

1. If your PR is not ready for reviews, but you want to run it on our CI, open a "Draft PR" to let us know you don't need feedback yet.

1. When you feel ready for integrating your work, mark your PR "Ready for review".

   - Your code should be readable and follow the project's design principles.
   - Make sure all tests are passing and any new code is tested for (coverage!).
   - Make sure you link the GitHub issue to your PR.
   - Make sure any docs for that piece of code are updated, or added.
   - The code should be elegant and simple. No over-engineering or hard-to-read code.

   Do your best but don't sweat about perfection! We do code-review to find any missed items.
   If you need help, don't hesitate to ping the core team on the PR.

1. Use tags in PR name for the following cases:

   - **\[blocked by #<number>\]** if your work is dependent on other PRs.
   - **\[wip\]** when you start to re-edit your work, mark it so no one will accidentally merge it in meantime.

### Question & Answer

#### How can I help/contribute?

All types of contributions are welcome - reporting bugs, fixing documentation, adding test cases, solving issues, and preparing bug fixes.
To get started with code contributions, look for issues marked with the label [good first issue](https://github.com/speediedan/finetuning-scheduler/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) or chose something close to your domain with the label [help wanted](https://github.com/speediedan/finetuning-scheduler/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). Before coding, make sure that the issue description is clear and comment on the issue so that we can assign it to you (or simply self-assign if you can).

#### Is there a recommendation for branch names?

We recommend you follow this convention `<type>/<issue-id>_<short-name>` where the types are: `bugfix`, `feature`, `docs`, or `tests` (but if you are using your own fork that's optional).

#### How to add new tests?

We are using [pytest](https://docs.pytest.org/en/stable/) with Finetuning Scheduler.

Here is the process to create a new test

- 0. Find a file in tests/ which match what you want to test. If none, create one.
- 1. Use this template to get started !
- 2. Use **BoringModel and derivatives to test out your code**.

```python
# TEST SHOULD BE IN YOUR FILE: tests/..../...py
# TEST CODE TEMPLATE

# [OPTIONAL] pytest decorator
# @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_explain_what_is_being_tested(tmpdir):
    """
    Test description about text reason to be
    """

    class ExtendedModel(BoringModel):
        ...

    model = ExtendedModel()

    # BoringModel is a functional model. You might want to set methods to None to test your behaviour
    # Example: model.training_step_end = None

    trainer = Trainer(default_root_dir=tmpdir, ...)  # will save everything within a tmpdir generated for this test
    trainer.fit(model)
    trainer.test()  # [OPTIONAL]

    # assert the behaviour is correct.
    assert ...
```

run our/your test with

```bash
python -m pytest tests/..../...py::test_explain_what_is_being_tested -v --capture=no
```
