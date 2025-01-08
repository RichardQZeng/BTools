from dask.distributed import Client, progress
import time
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description="A simple example script")
parser.add_argument('repetitions', type=int)
args = parser.parse_args()
repetitions = args.repetitions

client = Client(scheduler_file='dask_scheduler.json')
tic = time.time()

def slow_increment(x):
    time.sleep(1)
    return x + 1, str(datetime.now())

futures = client.map(slow_increment, range(34*repetitions))
progress(futures)

toc = time.time()
print(client)
print(f"\033[34mTime spent in {repetitions} repetitions: {toc-tic}\033[0m")
