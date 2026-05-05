"""Tee output utility for logging stdout to both console and file."""

import sys


class TeeOutput:
    """Duplicate stdout to both console and file."""
    
    def __init__(self, file_path):
        """Initialize with file path for logging."""
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', buffering=1)  # Line buffering
        
    def write(self, message):
        """Write to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        """Close the log file and restore stdout."""
        self.log.close()
        sys.stdout = self.terminal
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()