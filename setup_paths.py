"""
Path setup for alpha-beta-CROWN submodule.
Import this module before any alpha-beta-CROWN imports.

IMPORTANT: alpha-beta-CROWN's complete_verifier must come BEFORE project root
in sys.path to avoid naming conflicts (e.g., both have 'utils' module).
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ABCROWN_ROOT = os.path.join(_PROJECT_ROOT, 'alpha-beta-CROWN')
_ABCROWN_VERIFIER = os.path.join(_ABCROWN_ROOT, 'complete_verifier')
_AUTO_LIRPA = os.path.join(_ABCROWN_ROOT, 'auto_LiRPA')

# Add project root first (at end of list, lower priority)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Add alpha-beta-CROWN paths at front (higher priority)
# This ensures alpha-beta-CROWN's modules (like 'utils') are found before project's
for path in [_AUTO_LIRPA, _ABCROWN_ROOT, _ABCROWN_VERIFIER]:
    if path not in sys.path:
        sys.path.insert(0, path)
