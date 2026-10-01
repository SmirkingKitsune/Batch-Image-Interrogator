"""Headless backend for the opt-in Electron front end (./run.sh --electron).

Runs the existing core/ and interrogators/ without Qt, serves them to the
Electron window over a token-protected socket on 127.0.0.1, and exits when
the window closes.
"""

from bridge.app import start_bridge

__all__ = ["start_bridge"]
