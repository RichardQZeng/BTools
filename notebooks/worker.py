import concurrent.futures
import os
import psutil
import time

from tqdm import *


def worker_process(i):
    return i * i  # square the argument

def f(x):
    return f"pid: {psutil.Process().pid}"
    


def _foo(my_number):
   square = my_number * my_number
   time.sleep(1)
   return square 

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_process, i) for i in range(10)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
