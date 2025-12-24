import sys
import signal
from unittest.mock import patch

# Mock signal handling on Windows
original_signal = signal.signal

def patched_signal(signalnum, handler):
    try:
        return original_signal(signalnum, handler)
    except ValueError as e:
        if "signal only works in main thread" in str(e):
            # Silently ignore signal errors in worker threads
            return None
        raise

signal.signal = patched_signal