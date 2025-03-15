#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPODx: Wrapper script to run medical_grpo.py
"""

import os
import sys

if __name__ == "__main__":
    # Check for the API key in environment 
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("You can set it by running:")
        print("    export OPENAI_API_KEY=your_api_key_here")
        print("Patient simulation will use default conditions.")
    
    # Run the medical_grpo.py script directly
    import importlib.util
    import subprocess
    
    medical_grpo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_grpo.py")
    subprocess.run([sys.executable, medical_grpo_path], check=True)