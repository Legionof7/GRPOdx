#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation script for Verdict package.
Run this script to install Verdict and its dependencies.
"""

import os
import sys
import subprocess

def install_verdict():
    """Install the Verdict package and its dependencies."""
    print("Installing Verdict package...")
    
    try:
        # Try to import Verdict to check if it's already installed
        try:
            import verdict
            print("Verdict is already installed.")
            return True
        except ImportError:
            pass
        
        # Install Verdict using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "verdict"])
        
        # Verify installation
        import verdict
        print(f"Successfully installed Verdict {verdict.__version__}")
        return True
        
    except Exception as e:
        print(f"Error installing Verdict: {e}")
        return False

if __name__ == "__main__":
    success = install_verdict()
    sys.exit(0 if success else 1)