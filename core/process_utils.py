import subprocess
import sys
import os
import threading
import signal

from queue import Queue
from datetime import datetime
from subprocess import TimeoutExpired

def _enqueue_output(pipe, queue):
    """
    Reads lines from a subprocess pipe and puts them into a queue.
    This function is meant to be run in a separate thread.
    """
    try:
        # The iter(callable, sentinel) is a robust pattern for reading
        # lines from a binary stream until the stream closes.
        for line_bytes in iter(pipe.readline, b''):
            line_str = line_bytes.decode('utf-8', errors='replace').strip()
            if line_str:
                queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {line_str}")
    finally:
        pipe.close()

def run_command_async(command_args):
    """
    Runs a command in a subprocess and streams its output to a queue.
    Returns the Popen process object and the output queue.
    """
    creationflags = 0
    start_new_session = False
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        start_new_session = True

    process = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Redirect stderr to stdout for combined output
        # We read the stream in binary mode (by omitting text=True) for robustness.
        # The application-level flushing handler in download_data.py is the
        # primary mechanism for ensuring real-time output.
        creationflags=creationflags,
        start_new_session=start_new_session
    )
    output_queue = Queue()

    # Create and start a single thread to read the combined output stream
    reader_thread = threading.Thread(target=_enqueue_output, args=(process.stdout, output_queue))
    reader_thread.daemon = True
    reader_thread.start()
    return process, output_queue

def terminate_process_tree(process):
    """Terminates a process and its entire process tree robustly."""
    if process is None:
        return
    try:
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        # The process may have already finished or been killed.
        pass