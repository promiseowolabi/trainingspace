"""pytest wrapper: one test per check in checks.py. Run `pytest` in this folder."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common")]

import pytest  # noqa: E402

import checks  # noqa: E402
import tscheck  # noqa: E402
import gpt as module  # noqa: E402


@pytest.mark.parametrize("check", checks.CHECKS, ids=tscheck.check_id)
def test_check(check):
    check(module)
