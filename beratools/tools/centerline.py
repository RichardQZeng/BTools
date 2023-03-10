import time
from random import random
from time import sleep
from multiprocessing.pool import Pool
from numpy import mean
import json
import argparse


class OperationCancelledException(Exception):
    pass


def centerline(callback, in_line, in_cost_raster, line_radius, process_segments, out_center_line):
    # for x in kwargs:
    #    callback(x)

    execute()
    callback('Centerline tool done.')


# task executed in a worker process
def task(identifier):
    # report a message
    value = mean(identifier)
    print(f'Task {len(identifier)} with {value} executed', flush=True)

    # block for a moment
    sleep(value * 10)

    # return the generated value
    return value


# protect the entry point
def execute():
    # create and configure the process pool
    data = [[random() for n in range(100)] for i in range(100)]
    try:
        with Pool() as pool:
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(task, data):
                print(f'Got result: {result}', flush=True)
                if result > 0.9:
                    print('Pool terminated.')
                    raise OperationCancelledException()

    except OperationCancelledException:
        print("Operation cancelled")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    args = parser.parse_args()

    centerline(print, **args.input)

