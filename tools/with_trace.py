"""Run an upstream script with an on-demand Python stack dump (SIGUSR1)."""
import faulthandler
import runpy
import signal
import sys
from pathlib import Path
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=False)
script = sys.argv[1]
sys.path.insert(0, str(Path(script).resolve().parent))
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
