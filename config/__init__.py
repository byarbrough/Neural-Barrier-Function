"""
Configuration module for barrier function verification.
Extends alpha-beta-CROWN's ConfigHandler with barrier-specific arguments.
"""

# Setup paths first
import setup_paths  # noqa: F401

# Import upstream arguments and extend with barrier-specific options
from . import barrier_config

# Re-export Config and Globals with barrier extensions
Config = barrier_config.Config
Globals = barrier_config.Globals

__all__ = ["Config", "Globals"]
