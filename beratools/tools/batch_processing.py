import os
import json
import argparse

from PyQt5.QtWidgets import QDialog, QGridLayout, QPushButton
from beratools.widgets.batch_processing_dlg import *
from beratools_main import *
bt = BeraTools()


def batch_processing(callback, batch_tool_name, in_project, processes, verbose):
    proj_data = None
    proj_tool_name = None
    proj_tasks = None
    if in_project and os.path.exists(in_project):
        with open(in_project, 'r') as project_file:
            proj_data = json.load(project_file)

    dialog = BP_Dialog(proj_data['tool_api'])
    dialog.openCSV(in_project)

    dialog.exec()

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
