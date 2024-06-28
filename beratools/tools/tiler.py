import os
from qtpy.QtWidgets import QApplication, QDialog

from beratools.tools.common import *
from beratools.gui.map_window import MapWindow


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
        self.in_crs = None
        self.out_crs = None

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
            return_raster = clip_raster(self.in_chm, item['geometry'], self.tile_buffer, item['raster'])

            if not return_lines.empty and return_raster:
                cell_data = {
                    'in_line': item['line'].as_posix(),
                    'in_chm': item['raster'].as_posix()
                }
                tasks_list.append(cell_data)
                step += 1
                print('%{}'.format(step / len(self.clip_data) * 100))

        project_data['tasks'] = tasks_list
        with open(self.out_project, 'w') as project_file:
            json.dump(project_data, project_file, indent=4)

    def generate_cells(self):
        part_x = 0
        part_y = 0
        width = 0
        height = 0

        with(rasterio.open(self.in_chm)) as raster:
            self.boundary = raster.bounds
            width = raster.width
            height = raster.height
            self.in_crs = raster.crs

        if self.boundary:
            part_x = math.ceil(width / self.tile_size)
            part_y = math.ceil(height / self.tile_size)
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

            return True

        return False

    def generate_tiles_info(self):
        tiles_info = {'count': len(self.clip_data), 'dimension': self.tile_size}
        return tiles_info

    def cells_to_coord_list(self):
        self.out_crs = CRS('EPSG:4326')
        transformer = Transformer.from_crs(self.in_crs, self.out_crs)
        coords_list = []
        if self.clip_data:
            for item in self.clip_data:
                geom = item['geometry']
                coords = mapping(geom)['coordinates']
                if len(coords) > 0:
                    wgs84_coords = list(transformer.itransform(coords[0]))
                    wgs84_coords = [list(pt) for pt in wgs84_coords]
                    coords_list.append(wgs84_coords)

        # find bounds
        x = [pt[0] for polygon in coords_list for pt in polygon]
        y = [pt[1] for polygon in coords_list for pt in polygon]
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)

        center = [(x_min + x_max) / 2, (y_min + y_max) / 2]

        return coords_list, [[x_min, y_min], [x_max, y_max]], center

    def shapefile_to_coord_list(self):
        lines = read_geoms_from_shapefile(self.in_line)
        line_coords = []

        coords_list = []
        for line in lines:
            coords = line['coordinates']
            for pt in coords:
                coords_list.append(list(pt))

            line_coords.append(coords_list)

        return line_coords

    def execute(self):
        if self.generate_cells():
            coords_list, bounds, center = self.cells_to_coord_list()
            map_window = MapWindow()

            # add lines to map
            # lines = self.shapefile_to_coord_list()
            # map_window.add_polylines_to_map(lines, 'gray')

            # generate raster footprint and add to the map
            footprint = generate_raster_footprint(self.in_chm)

            # add AOI polygon and tile polygons
            map_window.set_tiles_info(self.generate_tiles_info())
            map_window.add_polygons_to_map('cells', coords_list, 'magenta')
            map_window.add_polygons_to_map('base', footprint, 'green')
            map_window.set_view(center, 10)
            # bounds = [[56.143426823080134, 111.1130415762259], [56.26141944093645, 110.63627702636289]]
            # map_window.fit_bounds(bounds)
            flag = map_window.exec()

            if flag != QDialog.Accepted:
                return

            self.create_out_file_name()
            self.save_clip_files()

            # save polygons to shapefile
            out_cells_file = self.proj_path.joinpath('cells.shp')
            schema = {
                'geometry': 'Polygon'
            }
            driver = 'ESRI Shapefile'
            out_line_file = fiona.open(out_cells_file, 'w', driver, schema, self.in_crs)
            for item in self.clip_data:
                feature = {
                    'geometry': mapping(item['geometry'])
                }
                out_line_file.write(feature)

            del out_line_file
            return


if __name__ == '__main__':
    # supress web engine logging
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-logging --log-level=3"

    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False

    tiling = Tiler(print, **args.input, processes=int(args.processes), verbose=verbose)
    tiling.execute()

    sys.exit(app.exec_())
