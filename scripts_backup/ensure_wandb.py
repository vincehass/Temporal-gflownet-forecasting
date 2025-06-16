#!/usr/bin/env python3
"""
Utility script to ensure W&B is enabled for all commands.

This script wraps other command-line calls and ensures that W&B logging 
is always enabled by adding the appropriate flags if they're missing.

Usage:
    python scripts/ensure_wandb.py [original command with args]

Example:
    python scripts/ensure_wandb.py python scripts/train.py --config-name base_config

The script will add the necessary W&B flags if they're not already present.
"""

import sys
import os
import argparse
import subprocess
import json
from pathlib import Path

# Default W&B settings
DEFAULT_ENTITY = "nadhirvincenthassen"
DEFAULT_PROJECT = "temporal-gfn"
WANDB_CONFIG_FILE = Path.home() / ".wandb_defaults.json"

def load_default_settings():
    """Load default W&B settings from config file if it exists."""
    if WANDB_CONFIG_FILE.exists():
        try:
            with open(WANDB_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read W&B config file: {e}")
    
    return {
        "entity": DEFAULT_ENTITY,
        "project": DEFAULT_PROJECT,
        "mode": "online"
    }

def save_default_settings(settings):
    """Save default W&B settings to config file."""
    try:
        with open(WANDB_CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save W&B config file: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ensure W&B is enabled for all commands')
    
    # W&B specific arguments
    parser.add_argument('--wandb-entity', type=str, help='W&B entity name')
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--wandb-name', type=str, help='W&B run name')
    parser.add_argument('--offline', action='store_true', help='Use W&B in offline mode')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable W&B for this run')
    parser.add_argument('--set-defaults', action='store_true', 
                       help='Save provided entity/project as defaults for future runs')
    
    # The rest of the arguments are the original command
    parser.add_argument('command', nargs='+', help='Original command to run')
    
    return parser.parse_args()

def modify_command(original_cmd, wandb_args, settings):
    """
    Modify the original command to ensure W&B is enabled.
    
    Args:
        original_cmd: List of command line arguments
        wandb_args: Parsed W&B specific arguments
        settings: Default W&B settings
    
    Returns:
        Modified command list
    """
    cmd = original_cmd.copy()
    
    # If we're explicitly disabling W&B, don't modify the command
    if wandb_args.disable_wandb:
        return cmd
    
    # Check if the command is a Python script
    is_python_cmd = (cmd[0] == "python" or cmd[0].endswith(".py")) 
    
    # For training.py, add the Hydra-style parameters
    if is_python_cmd and any("train.py" in arg for arg in cmd):
        # Check if use_wandb is already set
        if not any("use_wandb=" in arg for arg in cmd):
            cmd.append("use_wandb=true")
        
        # Add wandb entity if not already present
        if not any("wandb_entity=" in arg for arg in cmd):
            entity = wandb_args.wandb_entity or settings["entity"]
            cmd.append(f"wandb_entity={entity}")
        
        # Add wandb project if not already present
        if not any("wandb_project=" in arg for arg in cmd):
            project = wandb_args.wandb_project or settings["project"]
            cmd.append(f"wandb_project={project}")
        
        # Add wandb name if provided
        if wandb_args.wandb_name and not any("wandb_name=" in arg for arg in cmd):
            cmd.append(f"wandb_name={wandb_args.wandb_name}")
        
        # Set offline mode if requested
        if wandb_args.offline and not any("wandb_mode=" in arg for arg in cmd):
            cmd.append("wandb_mode=offline")
    
    # For other scripts that accept --use-wandb style args
    elif is_python_cmd:
        # Check if use_wandb flag is already present
        has_use_wandb = any(arg == "--use-wandb" for arg in cmd)
        
        if not has_use_wandb:
            cmd.append("--use-wandb")
        
        # Check for entity flag
        has_entity = any(arg == "--wandb-entity" for i, arg in enumerate(cmd) if i < len(cmd) - 1)
        if not has_entity:
            entity = wandb_args.wandb_entity or settings["entity"]
            cmd.extend(["--wandb-entity", entity])
        
        # Check for project flag
        has_project = any(arg == "--wandb-project" for i, arg in enumerate(cmd) if i < len(cmd) - 1)
        if not has_project:
            project = wandb_args.wandb_project or settings["project"]
            cmd.extend(["--wandb-project", project])
        
        # Check for name flag
        has_name = any(arg == "--wandb-name" for i, arg in enumerate(cmd) if i < len(cmd) - 1)
        if not has_name and wandb_args.wandb_name:
            cmd.extend(["--wandb-name", wandb_args.wandb_name])
        
        # Check for offline flag
        has_offline = any(arg == "--offline" for arg in cmd)
        if not has_offline and wandb_args.offline:
            cmd.append("--offline")
    
    # For shell scripts, add environment variables
    else:
        # Prepend environment variable settings
        env_vars = []
        
        if wandb_args.offline:
            env_vars.append("WANDB_MODE=offline")
        
        if wandb_args.wandb_entity or "entity" in settings:
            entity = wandb_args.wandb_entity or settings["entity"]
            env_vars.append(f"WANDB_ENTITY={entity}")
        
        if wandb_args.wandb_project or "project" in settings:
            project = wandb_args.wandb_project or settings["project"]
            env_vars.append(f"WANDB_PROJECT={project}")
        
        if wandb_args.wandb_name:
            env_vars.append(f"WANDB_NAME={wandb_args.wandb_name}")
        
        if env_vars:
            env_prefix = " ".join(env_vars) + " "
            return [env_prefix + cmd[0]] + cmd[1:]
    
    return cmd

def main():
    """Main function to process and execute the command."""
    args = parse_arguments()
    settings = load_default_settings()
    
    # Update default settings if requested
    if args.set_defaults:
        if args.wandb_entity:
            settings["entity"] = args.wandb_entity
        if args.wandb_project:
            settings["project"] = args.wandb_project
        if args.offline:
            settings["mode"] = "offline"
        
        save_default_settings(settings)
        print(f"Updated default W&B settings: {settings}")
    
    # Modify the command to include W&B flags
    original_cmd = args.command
    modified_cmd = modify_command(original_cmd, args, settings)
    
    # Print the modified command
    cmd_str = " ".join(modified_cmd)
    print(f"Executing: {cmd_str}")
    
    # Execute the modified command
    try:
        result = subprocess.run(modified_cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 