from shapely.geometry import shape


import fiona
import rasterio
import rasterio.mask

with fiona.open(r"D:\BERA_Tools\Rasterio\tests\data\bera.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

buffer = shape(shapes[0]).buffer(30)

with rasterio.open(r"D:\BERA_Tools\Rasterio\tests\data\RGB.byte.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, [buffer], crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)