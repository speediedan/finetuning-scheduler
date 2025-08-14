import time

_this_year = time.strftime("%Y")
__version__ = "2.5.3"
__author__ = "Dan Dale"
__author_email__ = "danny.dale@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2021-{_this_year}, {__author__}"
__homepage__ = "https://github.com/speediedan/finetuning-scheduler"
__docs_url__ = "https://finetuning-scheduler.readthedocs.io/en/latest/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "A PyTorch Lightning extension that enhances model experimentation with flexible fine-tuning schedules."
__long_docs__ = """
The FinetuningScheduler callback accelerates and enhances foundation model experimentation with flexible fine-tuning
schedules. Training with the FinetuningScheduler callback is simple and confers a host of benefits:

    - it dramatically increases fine-tuning flexibility
    - expedites and facilitates exploration of model tuning dynamics
    - enables marginal performance improvements of fine-tuned models

Fundamentally, the FinetuningScheduler callback enables multi-phase, scheduled fine-tuning of foundation models.
Gradual unfreezing (i.e. thawing) can help maximize foundation model knowledge retention while allowing (typically
upper layers of) the model to optimally adapt to new tasks during transfer learning.

FinetuningScheduler orchestrates the gradual unfreezing of models via a fine-tuning schedule that is either implicitly
generated (the default) or explicitly provided by the user (more computationally efficient). fine-tuning phase
transitions are driven by FTSEarlyStopping criteria (a multi-phase extension of EarlyStopping), user-specified epoch
transitions or a composition of the two (the default mode). A FinetuningScheduler training session completes when the
final phase of the schedule has its stopping criteria met.

Documentation
-------------
- https://finetuning-scheduler.readthedocs.io/en/latest/
- https://finetuning-scheduler.readthedocs.io/en/stable/
"""

__all__ = ["__author__", "__author_email__", "__copyright__", "__docs__", "__homepage__", "__license__", "__version__"]
