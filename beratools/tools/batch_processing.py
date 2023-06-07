import os
import csv
import argparse
import pathlib

from PyQt5.QtWidgets import QDialog, QGridLayout, QPushButton
from beratools.widgets.batch_processing_dlg import *
from beratools_main import *
bt = BeraTools()


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
        path_line = in_line.with_name(in_line.stem+'_outline.shp')
        path_footprint = in_line.with_name(in_line.stem + '_footprint.shp')
        path_canopy = in_chm.with_name(in_chm.stem+'_canopy.tif')
        path_cost = in_chm.with_name(in_chm.stem+'_canopy.tif')

        if tool_name == 'Canopy Cost Raster':
            task['in_chm'] = in_chm.as_posix()
            task['out_canopy'] = path_canopy.as_posix()
            task['out_cost'] = path_cost.as_posix()
        elif tool_name == 'Center Line':
            task['in_line'] = path_line.as_posix()
            task['in_cost'] = path_cost.as_posix()
            task['out_line'] = path_line.as_posix()
        elif tool_name == 'Line Footprint':
            task['in_line'] = path_line.as_posix()
            task['in_canopy'] = path_canopy.as_posix()
            task['in_cost'] = path_cost.as_posix()
            task['out_footprint'] = path_footprint.as_posix()
        elif tool_name == 'Dynamic CC and Line Footprint(Full)':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['out_footprint'] = in_line.with_name(in_line.stem + '_dyn_footprint.shp')
            task['out_line'] = in_line.with_name(in_line.stem + '_line_attributes.shp').as_posix()
        elif tool_name == 'Forest Line Attributes':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['in_footprint'] = path_footprint.as_posix()
            task['out_line'] = in_line.with_name(in_line.stem + '_line_attributes.shp').as_posix()
        elif tool_name == 'Raster Line Attributes':
            task['in_line'] = path_line.as_posix()
            task['in_chm'] = in_chm.as_posix()
            task['out_line'] = in_line.with_name(in_line.stem+'_raster_attributes.shp').as_posix()

        all_tasks.append(task)

    header = list(task.keys())

    with open(script_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_tasks)

    return script_path


def batch_processing(callback, batch_tool_name, in_project, processes, verbose):
    proj_data = None
    proj_tool_name = None
    proj_tasks = None
    if in_project and os.path.exists(in_project):
        with open(in_project, 'r') as project_file:
            proj_data = json.load(project_file)

    csv_file = create_tool_batch_csv(in_project, batch_tool_name, proj_data['tasks'])
    dialog = BP_Dialog(batch_tool_name)
    dialog.openCSV(csv_file)

    flag = dialog.exec()
    if flag == QDialog.Rejected:
        return
    elif flag == QDialog.Accepted:
        pass


    if proj_data:
        if 'tool_api' not in proj_data.keys() or 'tasks' not in proj_data.keys():
            callback('Project file corrupted, please check.')
            return
        else:
            proj_tool_name = proj_data['tool_api']
            proj_tasks = proj_data['tasks']

            steps = len(proj_tasks)
            step = 0
            for task in proj_tasks:
                if batch_tool_name != proj_tool_name:
                    task = generate_task_params(task)

                execute_task(batch_tool_name, task)
                step += 1
                callback('%{}'.format(step/steps*100))


def execute_task(tool_api, task):
    bt.run_tool(tool_api, task)


def generate_task_params(task):
    """
    When project tool_api is tiler, tool parameters are missing.
    This function will retrieve default arguments
    and generate new file names such as cost/canopy/centerline file names
    """
    updated_task = task
    return updated_task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    verbose = True if args.verbose == 'True' else False

    app = QApplication(sys.argv)
    batch_processing(print, **args.input, processes=int(args.processes), verbose=verbose)
    sys.exit(app.exec_())
