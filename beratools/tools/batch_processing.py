import os
import sys
import csv
import json
import pandas as pd
from pathlib import Path

from PyQt5 import QtWidgets
from beratools.gui.bt_data import BTData
import beratools.tools.common as bt_common
from beratools.gui.batch_processing_dlg import BPDialog

bt = BTData()


# TODO: Check input file existence
def create_tool_batch_csv(project, tool_name, tasks):
    tool_api = tool_name
    proj_path = Path(project)
    script_path = str(proj_path.with_name(proj_path.stem + '_' + tool_api.replace(' ', '_') + '.csv'))

    param_list = bt.get_bera_tool_parameters_list(tool_name)

    all_tasks = []
    for item in tasks:
        task = param_list
        in_line = Path(item['in_line'])
        in_chm = Path(item['in_chm'])
        path_line = in_line.with_name(in_line.stem + '_output_line.shp')
        path_footprint = in_line.with_name(in_line.stem + '_footprint.shp')
        path_canopy = in_chm.with_name(in_chm.stem + '_canopy.tif')
        path_cost = in_chm.with_name(in_chm.stem + '_cost.tif')

        # TODO: change to tool api
        if tool_name == 'Canopy Cost Raster':
            task['in_chm'] = in_chm.as_posix()
            task['out_canopy'] = path_canopy.as_posix()
            task['out_cost'] = path_cost.as_posix()
        elif tool_name == 'Center Line':
            task['in_line'] = in_line.as_posix()
            task['in_cost'] = path_cost.as_posix()
            task['out_line'] = path_line.as_posix()
        elif tool_name == 'Line Footprint by Static Threshold':
            task['in_line'] = path_line.as_posix()
            task['in_canopy'] = path_canopy.as_posix()
            task['in_cost'] = path_cost.as_posix()
            task['out_footprint'] = path_footprint.as_posix()
        elif tool_name == 'Line Footprint by Dynamic Threshold':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['out_footprint'] = in_line.with_name(in_line.stem + '_dyn_footprint.shp')
        elif tool_name == 'Forest Line Attributes':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['in_footprint'] = path_footprint.as_posix()
            task['out_line'] = in_line.with_name(in_line.stem + '_line_attributes.shp').as_posix()
        elif tool_name == 'Raster Line Attributes':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['out_line'] = in_line.with_name(in_line.stem + '_raster_attributes.shp').as_posix()

        all_tasks.append(task.copy())

    header = list(task.keys())

    with open(script_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_tasks)

    return script_path


def batch_processing(callback, batch_tool_name, in_project, processes, verbose):
    proj_data = None
    if in_project and os.path.exists(in_project):
        with open(in_project, 'r') as project_file:
            proj_data = json.load(project_file)

    csv_file = create_tool_batch_csv(in_project, batch_tool_name, proj_data['tasks'])
    dialog = BPDialog(batch_tool_name)
    dialog.open_csv(csv_file)

    flag = dialog.exec()

    # import tasks data
    data = pd.read_csv(csv_file)
    task_data = data.to_dict(orient='records')

    if flag == QtWidgets.QDialog.Accepted and task_data:
        steps = len(task_data)
        step = 0

        print('{} tasks are prepared'.format(steps))
        print('-----------------------------------')

        step = 0
        total_steps = len(task_data)
        for task in task_data:
            print('Starting task #{} ...'.format(step))
            print(' "PROGRESS_LABEL {} task {} of {}" '.format(batch_tool_name, step, total_steps), flush=True)
            print(' %{} '.format(step / steps * 100), flush=True)
            task = generate_task_params(task)
            code = execute_task(bt.get_bera_tool_api(batch_tool_name), task)
            step += 1
            # task is cancelled
            if code == 2:
                break

        print(' "PROGRESS_LABEL {} {} tasks finished" '.format(batch_tool_name, total_steps), flush=True)
        print(' %{} '.format(0), flush=True)

    print('Tasks finished.', flush=True)


def execute_task(tool_api, task):
    # return 2 if task is cancelled
    return bt.run_tool(tool_api, task, None, False)


def generate_task_params(task):
    """
    When project tool_api is tiler, tool parameters are missing.
    This function will retrieve default arguments
    and generate new file names such as cost/canopy/centerline file names
    """
    updated_task = task
    return updated_task


if __name__ == '__main__':
    in_args, in_verbose = bt_common.check_arguments()
    app = QtWidgets.QApplication(sys.argv)
    batch_processing(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    sys.exit(app.exec_())
