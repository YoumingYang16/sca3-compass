"""Owned small process fixture for supervisor timeout, not a research experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from research_window import wait_start_gate,cooperative_stop,write

wait_start_gate()
parser=argparse.ArgumentParser();parser.add_argument('--ignore-stop',action='store_true');args=parser.parse_args()
out=Path(os.environ['RESEARCH_STOP_FILE']).parent
if args.ignore_stop:
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
    write(out/'descendant.json',{'pid':child.pid,'owned_by_fixture':True})
for i in range(1200):
    write(out/'checkpoint.json',{'tick':i,'stopped':False})
    if cooperative_stop() and not args.ignore_stop:
        write(out/'checkpoint.json',{'tick':i,'stopped':True});break
    time.sleep(.1)
