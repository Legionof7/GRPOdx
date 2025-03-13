#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Proper setup script for Verdict package.
This file ensures Verdict is properly installed in the environment.
"""

import sys
import subprocess

def install_verdict():
    """Install the Verdict package if not already installed."""
    print("Checking for Verdict installation...")
    
    try:
        # Try to import Verdict to check if it's already installed
        import verdict
        print(f"Verdict is already installed (version {verdict.__version__}).")
        return True
    except ImportError:
        print("Verdict is not installed. Installing now...")
        
    # Install Verdict using pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "verdict"])
        
        # Verify installation
        import verdict
        print(f"Successfully installed Verdict {verdict.__version__}")
        return True
    except Exception as e:
        print(f"Error installing Verdict: {e}")
        return False

if __name__ == "__main__":
    install_verdict()