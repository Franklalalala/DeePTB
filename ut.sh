#!/bin/sh
conda init
source activate deeptb
pip install ".[test]"
# `python -m pytest` (not the bare console script) puts the repo root on
# sys.path, which the test modules importing the repo-root `tools` package need.
python -m pytest ./dptb/tests/
