import warnings
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing.pool import Pool

import pandas as pd
import geopandas as gpd

import dask.distributed as dask_dist
from dask import config as dask_cfg
# import ray

import beratools.core.constants as bt_const

# settings for dask
dask_cfg.set({"distributed.scheduler.worker-ttl": None})
warnings.simplefilter("ignore", dask_dist.comm.core.CommClosedError)
warnings.simplefilter(action="ignore", category=FutureWarning)


class ToolBase(object):
    def __init__(self):
        pass

    def execute_multiprocessing(self):
        pass


def result_is_valid(result):
    if type(result) is list or type(result) is tuple:
        if len(result) > 0:
            return True
    elif (
        type(result) is pd.DataFrame
        or type(result) is gpd.GeoDataFrame
        or type(result) is pd.Series
        or type(result) is gpd.GeoSeries
    ):
        if not result.empty:
            return True
    elif result:
        return True

    return False


def print_msg(app_name, step, total_steps):
    print(f' "PROGRESS_LABEL {app_name} {step} of {total_steps}" ', flush=True)
    print(f' %{step / total_steps * 100} ', flush=True)


def execute_multiprocessing(
    in_func,
    in_data,
    app_name,
    processes,
    workers=1,
    mode=bt_const.PARALLEL_MODE,
    verbose=False,
    scheduler_file="dask_scheduler.json",
):
    out_result = []
    step = 0
    print("Using {} CPU cores".format(processes), flush=True)
    total_steps = len(in_data)

    try:
        if mode == bt_const.ParallelMode.MULTIPROCESSING:
            print("Multiprocessing started...", flush=True)

            with Pool(processes) as pool:
                # print(multiprocessing.active_children())
                with tqdm(total=total_steps, disable=verbose) as pbar:
                    for result in pool.imap_unordered(in_func, in_data):
                        if result_is_valid(result):
                            out_result.append(result)

                        step += 1
                        if verbose:
                            print_msg(app_name, step, total_steps)
                        else:
                            pbar.update()

            pool.close()
            pool.join()
        elif mode == bt_const.ParallelMode.SEQUENTIAL:
            with tqdm(total=total_steps, disable=verbose) as pbar:
                for line in in_data:
                    result_item = in_func(line)
                    if result_is_valid(result_item):
                        out_result.append(result_item)

                    step += 1
                    if verbose:
                        print_msg(app_name, step, total_steps)
                    else:
                        pbar.update()
        elif mode == bt_const.ParallelMode.CONCURRENT:
            with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
                futures = [executor.submit(in_func, line) for line in in_data]
                with tqdm(total=total_steps, disable=verbose) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        result_item = future.result()
                        if result_is_valid(result_item):
                            out_result.append(result_item)

                        step += 1
                        if verbose:
                            print_msg(app_name, step, total_steps)
                        else:
                            pbar.update()
        elif mode == bt_const.ParallelMode.DASK:
            dask_client = dask_dist.Client(threads_per_worker=1, n_workers=processes)
            print(f"Local Dask client: {dask_client}")
            try:
                print('start processing')
                result = dask_client.map(in_func, in_data)
                seq = dask_dist.as_completed(result)

                with tqdm(total=total_steps, disable=verbose) as pbar:
                    for i in seq:
                        if result_is_valid(result):
                            out_result.append(i.result())

                        step += 1
                        if verbose:
                            print_msg(app_name, step, total_steps)
                        else:
                            pbar.update()
            except Exception as e:
                dask_client.close()

            dask_client.close()
        elif mode == bt_const.ParallelMode.SLURM:
            dask_client = dask_dist.Client(scheduler_file=scheduler_file)
            print(f"Slurm cluster Dask client: {dask_client}")
            try:
                print("start processing")
                result = dask_client.map(in_func, in_data)
                seq = dask_dist.as_completed(result)
                dask_dist.progress(result)

                for i in seq:
                    if result_is_valid(result):
                        out_result.append(i.result())
            except Exception as e:
                dask_client.close()

            dask_client.close()
        # ! important !
        # comment temporarily, man enable later if need to use ray
        # elif mode == bt_const.ParallelMode.RAY:
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
    except Exception as e:
        print(e)
        return None

    return out_result
