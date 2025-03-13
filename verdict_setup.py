#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to install Verdict package.
"""

import sys
import subprocess

# Just directly install Verdict - no checks or complex logic
subprocess.check_call([sys.executable, "-m", "pip", "install", "verdict"])
print("Verdict installation complete.")