#!/usr/bin/env python3
"""Run this lab's checks and print a completion code when they all pass.

    PREFIX=<your-handle> python verify.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import tscheck  # noqa: E402

LAB = "ddp-from-scratch"
MODULE = "ddp"

if __name__ == "__main__":
    sys.exit(tscheck.main(LAB, MODULE, HERE))
