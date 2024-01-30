import shapely
from shapely.geometry import mapping, shape, Polygon, MultiPolygon
import fiona
import alphashape
from label_centerlines import get_centerline

src_shp = r"D:\BT_Test\ConcaveHull\footprint_fixed.shp"
dst_shp = r"D:\BT_Test\ConcaveHull\footprint_no_holes_simp.shp"
line_shp = r"D:\BT_Test\ConcaveHull\centerline_no_holes_simp.shp"

pts_list = []
poly_list = []
dst_crs = None
single_poly = []

with fiona.open(src_shp) as src:
    src_schema = src.schema
    dst_crs = src.crs
    for feat in src:
        geom = shape(feat.geometry)
        poly = shapely.segmentize(geom, max_segment_length=10)

        exterior_pts = []
        if type(poly) == MultiPolygon:
            single_poly.append(False)
            for i in poly.geoms:
                exterior_pts += list(i.exterior.coords)
        elif type(poly) == Polygon:
            single_poly.append(True)
            exterior_pts = list(poly.exterior.coords)
            poly = Polygon(exterior_pts)
            poly = poly.simplify(1)

        poly_list.append(poly)
        pts_list.append(exterior_pts)

dst_geoms = []

for index, pt_list, poly in zip(enumerate(single_poly), pts_list, poly_list):
    # alpha = alphashape.optimizealpha(i)
    alpha = 0.05
    i = index[0]
    single = index[1]
    if single:
        alpha_shp = poly
    else:
        alpha_shp = alphashape.alphashape(pt_list, alpha)

    dst_geoms.append(alpha_shp)

# use shapely concave_hull
# alphashape has better output
# shapely: 0.1, alphashape: 0.03
# for i in poly_list:
#     alpha_shp = shapely.concave_hull(i, ratio=0.1)
#     dst_geoms.append(alpha_shp)

# generate centerlines
centerlines = []
for i, poly in enumerate(dst_geoms):
    line = get_centerline(poly, segmentize_maxlen=1, max_points=3000, simplification=0.05, smooth_sigma=0.5, max_paths=1)
    centerlines.append(line)
    print('Polygon {} done'.format(i))

# Write out concave polygons
dst_schema = {
    'geometry': 'Polygon',
    'properties': {},
}

with fiona.open(dst_shp, mode='w', driver='ESRI Shapefile', crs=dst_crs, schema=dst_schema) as c:
    for poly in dst_geoms:
        c.write({

            'geometry': mapping(poly),
            'properties': {},
        })

# Write out centerlines
line_schema = {
    'geometry': 'LineString',
    'properties': {},
}

# Write a new Shapefile
with fiona.open(line_shp, mode='w', driver='ESRI Shapefile', crs=dst_crs, schema=line_schema) as c:
    for line in centerlines:
        c.write({
            'geometry': mapping(line),
            'properties': {},
        })