from multiprocessing.pool import Pool
import multiprocessing
import concurrent.futures
import warnings

import pandas as pd
import geopandas as gpd

from beratools.core.constants import *

# from dask.distributed import Client, as_completed
# from dask import config as cfg
# import dask.distributed
# import ray

# settings for dask
# cfg.set({'distributed.scheduler.worker-ttl': None})
# warnings.simplefilter("ignore", dask.distributed.comm.core.CommClosedError)


class OperationCancelledException(Exception):
    pass


class ToolBase(object):
    def __init__(self):
        pass

    def execute_multiprocessing(self):
        pass


def result_is_valid(result):
    if type(result) is list or type(result) is tuple:
        if len(result) > 0:
            return True
    elif type(result) is pd.DataFrame or type(result) is gpd.GeoDataFrame:
        if not result.empty:
            return True
    elif result:
        return True

    return False


def print_msg(app_name, step, total_steps):
    print(f' "PROGRESS_LABEL {app_name} {step} of {total_steps}" ', flush=True)
    print(f' %{step / total_steps * 100} ', flush=True)


def execute_multiprocessing(in_func, in_data, app_name, processes, workers,
                            mode=PARALLEL_MODE, verbose=False):
    out_result = []
    step = 0
    print("Using {} CPU cores".format(processes))
    total_steps = len(in_data)

    try:
        if mode == ParallelMode.MULTIPROCESSING:
            multiprocessing.set_start_method('spawn')
            print("Multiprocessing started...")

            with Pool(processes) as pool:
                print(multiprocessing.active_children())
                for result in pool.imap_unordered(in_func, in_data):
                    if result_is_valid(result):
                        out_result.append(result)

                    step += 1
                    print_msg(app_name, step, total_steps)

            pool.close()
            pool.join()
        elif mode == ParallelMode.SEQUENTIAL:
            for line in in_data:
                result_item = in_func(line)
                if result_is_valid(result_item):
                    out_result.append(result_item)

                step += 1
                print_msg(app_name, step, total_steps)
        elif mode == ParallelMode.CONCURRENT:
            with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
                futures = [executor.submit(in_func, line) for line in in_data]
                for future in concurrent.futures.as_completed(futures):
                    result_item = future.result()
                    if result_is_valid(result_item):
                        out_result.append(result_item)

                    step += 1
                    print_msg(app_name, step, total_steps)

        # ! important !
        # comment temporarily, man enable later if need to use dask or ray
        # elif mode == ParallelMode.DASK:
        #     dask_client = Client(threads_per_worker=1, n_workers=processes)
        #     print(dask_client)
        #     try:
        #         print('start processing')
        #         result = dask_client.map(in_func, in_data)
        #         seq = as_completed(result)
        #
        #         for i in seq:
        #             if result_is_valid(result):
        #                 out_result.append(i.result())
        #
        #             step += 1
        #             print_msg(app_name, step, total_steps)
        #     except Exception as e:
        #         dask_client.close()
        #
        #     dask_client.close()
        # elif mode == ParallelMode.RAY:
        #     ray.init(log_to_driver=False)
        #     process_single_line_ray = ray.remote(in_func)
        #     result_ids = [process_single_line_ray.remote(item) for item in in_data]
        #
        #     while len(result_ids):
        #         done_id, result_ids = ray.wait(result_ids)
        #         result_item = ray.get(done_id[0])
        #
        #         if result_is_valid(result_item):
        #             out_result.append(result_item)
        #
        #         step += 1
        #         print_msg(app_name, step, total_steps)

        #     ray.shutdown()
    except OperationCancelledException:
        print("Operation cancelled")
        return None

    return out_result
