import geopandas as gpd
import shapely
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union

shp_in = r"D:\BT_Test\ConcaveHull\corridor_centerline_smooth-2.shp"
shp_out = r"D:\BT_Test\ConcaveHull\corridor_centerline_smooth-2_snapped.shp"

BUFFER_CLIP = 2.5
BUFFER_CENTROID = 3
BUFFER_QUERY = 10

FIND_NEW_VERTEX = True


def snap_line_grp(line_index, pt_index, spatial_index):
    line = geom[index]
    if len(line.coords) <= 1:
        return

    pt = Point(line.coords[pt_index])
    pt_buffer = pt.buffer(BUFFER_QUERY)
    line_buffer = line.buffer(BUFFER_QUERY)
    lines = spatial_index.query(line_buffer)

    idx_dicts = [(line_index, pt_index)]

    # remove current line
    idx_set = set(lines)
    idx_set.remove(line_index)
    lines = list(idx_set)

    if len(lines) > 0:
        for idx in lines:
            line = geom[idx]
            if len(line.coords) <= 1:
                continue
            if pt_buffer.contains(Point(line.coords[0])):
                idx_dicts.append((idx, 0))
            elif pt_buffer.contains(Point(line.coords[-1])):
                idx_dicts.append((idx, -1))

    ploy_list = []
    if len(idx_dicts) > 1:
        for idx in idx_dicts:
            line = geom[idx[0]]
            end_buffer = Point(line.coords[idx[1]]).buffer(BUFFER_CLIP)
            pt_clip = end_buffer.exterior.intersection(line)
            centroid_buffer = pt_clip.buffer(BUFFER_CENTROID)

            line = line.difference(centroid_buffer)

            geom[idx[0]] = line
            ploy_list.append(centroid_buffer)

    new_vertex = None
    if ploy_list
       if FIND_NEW_VERTEX:
            new_vertex = unary_union(ploy_list).centroid
       else:  # use least cost path intersection
           pass
    else:
        return

    if not new_vertex:
        return

    for idx in idx_dicts:
        line = geom[idx[0]]
        coords = list(line.coords)

        idx_vertex = idx[1]
        if idx_vertex == 0:
            coords.insert(0, new_vertex)
        elif idx_vertex == -1:
            coords.append(new_vertex)

        if coords:
            if len(coords) >= 2:
                geom[idx[0]] = LineString(coords)


data = gpd.read_file(shp_in)
geom = data.geometry
sindex = data.sindex

# c = shapely.affinity.rotate(a, 180, origin=pt1)

end_pt_processed = [{0: False, -1: False}] * len(geom)

for index, i in enumerate(end_pt_processed):
    if not i[0]:
        snap_line_grp(index, 0, sindex)
        end_pt_processed[index] = True

    if not i[-1]:
        snap_line_grp(index, -1, sindex)
        end_pt_processed[index] = True

    print('line {}'.format(index))

geom.to_file(shp_out)