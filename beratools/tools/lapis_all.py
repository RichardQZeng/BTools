
from random import random
import time
from multiprocessing.pool import Pool
from numpy import mean
from tools.common import *
import json


class OperationCancelledException(Exception):
    pass


def lapis_all(args, callback=print, processes=None, verbose=None):
    lapis_path = '../third_party/Lapis_0_8/Lapis.exe'
    lapis_path = Path(__file__).parent.joinpath(lapis_path).resolve()
    ini_file = Path(__file__).parents[2].joinpath(r'.\.data\lapis.ini').resolve().as_posix()

    arg_parsed = json.loads(args)
    in_dtm = arg_parsed['in_dtm']
    in_las = arg_parsed['in_las']
    out_dir = arg_parsed['out_dir']

    f = open(ini_file, 'w')
    f.write('#Data options\n')
    f.write('dem=' + in_dtm + '\n')
    f.write('dem-units = unspecified' + '\n')
    f.write('dem-algo = raster' + '\n')
    f.write('las=' + in_las + '\n')
    f.write('las-units=unspecified' + '\n')
    f.write('output=' + out_dir + '\n')

    f.write('\n # Computer-specific options\n')
    f.write('thread=50' + '\n')
    f.write('bench=' + '\n')

    f.write('\n # Processing options\n')
    f.write('xres=0.15' + '\n')
    f.write('yres=0.15' + '\n')
    f.write('xorigin=0' + '\n')
    f.write('yorigin=0' + '\n')
    f.write('csm-cellsize=0.15' + '\n')
    f.write('footprint=0.1' + '\n')
    f.write('smooth=1' + '\n')
    f.write('minht=-8' + '\n')
    f.write('maxht=100' + '\n')
    f.write('class=~7, 9, 18' + '\n')
    f.write('max-scan-angle=32' + '\n')
    f.write('user-units=meters' + '\n')
    f.write('canopy=3' + '\n')
    f.write('strata=0.5, 1, 2, 4, 8, 16, 32, 48, 64,' + '\n')
    f.write('min-tao-dist=1' + '\n')
    f.write('id-algo=highpoint' + '\n')
    f.write('seg-algo=watershed' + '\n')
    f.write('topo-scale=500, 1000, 2000,' + '\n')
    f.write('fine-int=' + '\n')

    args = lapis_path.as_posix() + ' --ini-file' + ' ' + ini_file

    callback('Lapis parameters returned.')
    return args


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
    try:
        total_steps = 300
        with Pool() as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(worker, data):
                print(f'Got result: {result}', flush=True)
                step += 1
                print(step)
                print('%{}'.format(step/total_steps*100))

    except OperationCancelledException:
        print("Operation cancelled")


if __name__ == '__main__':
    # in_args, in_verbose = check_arguments()
    # start_time = time.time()
    # lapis_all(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    #
    # print('Elapsed time: {}'.format(time.time() - start_time))

    pass
