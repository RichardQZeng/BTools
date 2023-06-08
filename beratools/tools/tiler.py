import argparse
import json
import math
import fiona
from pathlib import Path
from shapely.geometry import shape, Polygon, mapping

from common import *


class Tiler:
    def __init__(self, callback, in_line, in_chm, tile_size,
                 tile_buffer, out_project, processes, verbose):

        self.in_line = in_line
        self.in_chm = in_chm
        self.boundary = None
        self.tile_size = float(tile_size)
        self.tile_buffer = float(tile_buffer)
        self.out_project = out_project
        self.processes = processes
        self.verbose = verbose
        self.clip_data = []
        self.proj_path = ''

    def create_out_file_name(self):
        # prepare path
        # TODO: more formats later, now only tiff and shp are considered
        path_chm = Path(self.in_chm)
        path_line = Path(self.in_line)
        self.proj_path = path_chm.parent
        path_root = self.proj_path.joinpath('cells')

        if not path_root.exists():
            path_root.mkdir()

        item_count = len(self.clip_data)
        str_len = len(str(item_count))
        for i, item in enumerate(self.clip_data):
            cell_num = str(i).zfill(str_len)
            self.clip_data[i]['raster'] = path_root.joinpath(path_chm.stem + '_' + cell_num + '.tif')
            self.clip_data[i]['line'] = path_root.joinpath(path_line.stem + '_' + cell_num + '.shp')

    def save_clip_files(self):
        project_data = {'tool_api': 'tiler'}
        tasks_list = []
        step = 0

        print('Generating {} tiles ...'.format(len(self.clip_data)))
        for item in self.clip_data:
            return_lines = clip_lines(item['geometry'], self.tile_buffer, self.in_line, item['line'])
            return_raster = clip_raster(item['geometry'], self.tile_buffer, self.in_chm, item['raster'])

            if not return_lines.empty and return_raster:
                cell_data = {
                    'in_line': item['line'].as_posix(),
                    'in_chm': item['raster'].as_posix()
                }
                tasks_list.append(cell_data)
                step += 1
                print('%{}'.format(step/len(self.clip_data)*100))

        project_data['tasks'] = tasks_list
        with open(self.out_project, 'w') as project_file:
            json.dump(project_data, project_file, indent=4)

    def execute(self):
        part_x = 0
        part_y = 0
        width = 0
        height = 0

        in_crs = None
        with(rasterio.open(self.in_chm)) as raster:
            self.boundary = raster.bounds
            width = raster.width
            height = raster.height
            in_crs = raster.crs

        if self.boundary:
            part_x = math.ceil(width/self.tile_size)
            part_y = math.ceil(height/self.tile_size)
            min_x, min_y, max_x, max_y = self.boundary
            polygon_bound = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

            step_x = (max_x - min_x) / part_x
            step_y = (max_y - min_y) / part_y
            cells = []
            for i in range(part_x):
                for j in range(part_y):
                    cells.append(Polygon([(min_x + i * step_x, min_y + j * step_y),
                                          (min_x + (i + 1) * step_x, min_y + j * step_y),
                                          (min_x + (i + 1) * step_x, min_y + (j + 1) * step_y),
                                          (min_x + i * step_x, min_y + (j + 1) * step_y)]))

            # remove polygons not in boundary
            for cell in cells:
                if not polygon_bound.disjoint(cell):
                    self.clip_data.append({'geometry': cell})

            self.create_out_file_name()
            self.save_clip_files()

            # save polygons to shapefile
            out_cells_file = self.proj_path.joinpath('cells.shp')
            schema = {
                'geometry': 'Polygon'
            }
            driver = 'ESRI Shapefile'
            out_line_file = fiona.open(out_cells_file, 'w', driver, schema, in_crs)
            for item in self.clip_data:
                feature = {
                    'geometry': mapping(item['geometry'])
                }
                out_line_file.write(feature)

            del out_line_file
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False

    tiling = Tiler(print, **args.input, processes=int(args.processes), verbose=verbose)
    tiling.execute()
