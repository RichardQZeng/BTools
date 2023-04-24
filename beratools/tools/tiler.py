import argparse
import json
import fiona
from pathlib import Path
from shapely.geometry import shape, Polygon, mapping

from common import *


class Tiler:
    def __init__(self, callback,
                 in_line, in_chm, in_boundary, tile_size, tile_buffer, out_project,
                 processes, verbose):

        self.in_line = in_line
        self.in_chm = in_chm
        self.in_boundary = in_boundary
        self.tile_size = float(tile_size)
        self.tile_buffer = float(tile_buffer)
        self.out_project = out_project
        self.processes = processes
        self.verbose = verbose
        self.clip_data = []

    def create_out_file_name(self):
        # prepare path
        # TODO: more formats later, now only tiff and shp are considered
        path_chm = Path(self.in_chm)
        path_line = Path(self.in_line)
        path_root = path_chm.parent.joinpath('cells')

        if not path_root.exists():
            path_root.mkdir()

        item_count = len(self.clip_data)
        str_len = len(str(item_count))
        for i, item in enumerate(self.clip_data):
            cell_num = str(i).zfill(str_len)
            self.clip_data[i]['raster'] = path_root.joinpath(path_chm.stem + '_' + cell_num + '.tif')
            self.clip_data[i]['line'] = path_root.joinpath(path_line.stem + '_' + cell_num + '.shp')

    def save_clip_files(self):
        project_data = {'tool': 'tiler'}
        tasks_list = []
        step = 0
        for item in self.clip_data:
            clip_lines(item['geometry'], self.tile_buffer, self.in_line, item['line'])
            clip_raster(item['geometry'], self.tile_buffer, self.in_chm, item['raster'])

            cell_data = {
                'in_line': self.in_line,
                'in_chm': self.in_chm
            }
            tasks_list.append(cell_data)
            step += 1
            print('%{}'.format(step/len(self.clip_data)*100))

        project_data['tasks'] = tasks_list
        with open(self.out_project, 'w') as project_file:
            json.dump(project_data, project_file, indent=4)

    def execute(self):
        part_x = 5
        part_y = 3
        if self.in_boundary:
            polygons = []
            in_crs = None
            with fiona.open(self.in_boundary) as boundary_file:
                in_crs = boundary_file.crs
                for record in boundary_file:
                    if record.geometry.type == "Polygon" or record.geometry.type == "MultiPolygon":
                        polygons.append(record.geometry)

            if len(polygons) <= 0:
                print("No polygons found in boundary file.")
                return
            # TODO: multipolygon need tests

            boundary = shape(polygons[0])
            minx, miny, maxx, maxy = boundary.bounds
            step_x = (maxx - minx) / part_x
            step_y = (maxy - miny) / part_y
            cells = []
            for i in range(part_x):
                for j in range(part_y):
                    cells.append(Polygon([(minx + i * step_x, miny + j * step_y),
                                          (minx + (i + 1) * step_x, miny + j * step_y),
                                          (minx + (i + 1) * step_x, miny + (j + 1) * step_y),
                                          (minx + i * step_x, miny + (j + 1) * step_y)]))

            # remove polygons not in boundary
            for cell in cells:
                if not boundary.disjoint(cell):
                    self.clip_data.append({'geometry': cell})

            self.create_out_file_name()
            self.save_clip_files()

            # save polygons to shapefile
            # out_polygon_file = r'D:\Temp\Tesspy\cell.shp'
            # schema = {
            #     'geometry': 'Polygon'
            # }
            # driver = 'ESRI Shapefile'
            # out_line_file = fiona.open(out_polygon_file, 'w', driver, schema, in_crs)
            # for cell in cells_intersected:
            #     feature = {
            #         'geometry': mapping(cell['geometry'])
            #     }
            #     out_line_file.write(feature)
            #
            # del out_line_file

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
