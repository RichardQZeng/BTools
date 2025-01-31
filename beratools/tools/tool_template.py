import time
from multiprocessing.pool import Pool
from random import random

from common import *
from numpy import mean


def tool_name(callback, in_line, in_cost_raster, line_radius, process_segments, out_center_line):
    execute_multiprocessing()
    callback('tool_template tool done.')


# task executed in a worker process
def worker(task_data):
    # report a message
    value = mean(task_data)
    print(f'Task {len(task_data)} with {value} executed', flush=True)

    # block for a moment
    time.sleep(value * 10)

    # return the generated value
    return value


# protect the entry point
def execute_multiprocessing():
    # create and configure the process pool
    data = [[random() for n in range(100)] for i in range(300)]
    total_steps = 300
    with Pool() as pool:
        step = 0
        # execute tasks in order, process results out of order
        for result in pool.imap_unordered(worker, data):
            print(f'Got result: {result}', flush=True)
            step += 1
            print(step)
            print('%{}'.format(step / total_steps * 100))


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    tool_name(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
