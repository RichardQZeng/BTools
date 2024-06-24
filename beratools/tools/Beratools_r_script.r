chm2trees <- function(in_chm, Min_ws, hmin, out_folder, rprocesses)
{
    library(lidR)
    library(terra)
    library(future)


    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    current_chm <- rast(in_chm)
    cell_size <- res(current_chm)[1]
    # find the highest point of CHM
    tallest_ht = minmax(current_chm)[2]

    #Reforestation Standard of Alberta 2018
    #(https://www1.agric.gov.ab.ca/$department/deptdocs.nsf/all/formain15749/$FILE/reforestation-standard-alberta-may1-2018.pdf, p.53)
    #Live crown ratio is the proportion of total stem length that is covered by living branches. It is expressed as a percentage or decimal of the total tree height. Live crown ratio is a useful indicator of the status of the tree in relation to vigor, photosynthetic leaf area, and is inversely related to stocking density. It is assumed that live crown ratio must be greater than 0.3 (30%) in order for the tree to release well

    if (Min_ws >= (0.3 * hmin)) {
        (Min_ws <- Min_ws) }else {
        (Min_ws <- 0.3 * hmin) }

    f <- function(x) {
        y <- (x * 0.3) + Min_ws
        y[x < hmin] <- (Min_ws) # Smallest Crown
        y[x > tallest_ht] <- (tallest_ht * 0.3) # Largest Crown
        return(y)
    }

    out_ttop_filename = paste0(out_folder, "/", substr(basename(in_chm), 1, nchar(basename(in_chm)) - 4), ".shp")

    ttop <- locate_trees(current_chm, lmf(ws = f, hmin = hmin, shape = "circular"), uniqueness = "bitmerge")

    x <- vect(ttop)
    writeVector(x, out_ttop_filename, overwrite = TRUE)
    #st_write(ttop,out_ttop_filename)

}

##################################################################################################################
#create a 'generate_pd' function
generate_pd <- function(ctg, radius_fr_CHM, focal_radius, cell_size, cache_folder,
                        cut_ht, PD_Ground_folder, PD_Total_folder, rprocesses) {
    library(terra)
    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    opts <- paste0("-drop_class 7")

    print("Processing using R packages.")

    folder <- paste0(cache_folder, "/nlidar/n_{*}")
    opt_output_files(ctg) <- opt_output_files(ctg) <- folder
    opt_laz_compression(ctg) <- FALSE
    opt_filter(ctg) <- "-drop_class 7"
    opt_chunk_alignment(ctg) <- c(0, 0)

    catalog_laxindex = function(ctg)
    {
        stopifnot(is(ctg, "LAScatalog"))

        opt_chunk_size(ctg) <- 0
        opt_chunk_buffer(ctg) <- 0
        opt_wall_to_wall(ctg) <- FALSE
        opt_output_files(ctg) <- ""

        create_lax_file = function(cluster) {
            rlas::writelax(cluster@files)
            return(0)
        }

        options <- list(need_buffer = FALSE, drop_null = FALSE)

        catalog_apply(ctg, create_lax_file, .options = options)
        return(invisible())
    }

    #normalized LAS with pulse info
    print("Indexing LAS Tiles...")
    catalog_laxindex(ctg)
    print("Normalize point cloud using K-nearest neighbour IDW ...")
    normalize_height(ctg, algorithm = knnidw())

    print("Generating point density (total focal sum) raster ...")

    pd_total <- function(chunk, radius, cell_size)
    {
        las <- readLAS(chunk)
        if (is.empty(las)) return(NULL)

        las_1 <- filter_poi(readLAS(chunk), buffer == 0)
        hull <- st_convex_hull(las_1)
        # bbox <- ext(las_1)

        # convert to SpatialPolygons
        bbox <- vect(hull)

        las <- filter_poi(las, Classification != 7L)
        #las <- retrieve_pulses(las)
        density_raster_total <- rasterize_density(las, res = cell_size, pkg = "terra")[[1]]

        tfw <- focalMat(density_raster_total, radius, "circle")

        tfw[tfw > 0] = 1
        tfw[tfw == 0] = NA

        Total_focal <- focal(density_raster_total, w = tfw, fun = "sum", na.rm = TRUE, na.policy = "omit", fillvalue = NA, expand = FALSE)
        Total_focal <- crop(Total_focal, bbox)
    }

    opt <- list(need_output_file = TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg) <- c(0, 0)
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    opt_output_files(ctg) <- paste0(PD_Total_folder, "/{*}_PD_Tfocalsum")
    opt_stop_early(ctg) <- FALSE
    catalog_apply(ctg, pd_total, radius = focal_radius, cell_size = cell_size, .options = opt)

    #load normalized LAS for ground point density
    ht <- paste0("-drop_class 7 -drop_z_above ", cut_ht)
    ctg2 <- readLAScatalog(paste0(cache_folder, "/nlidar"), filter = ht)
    catalog_laxindex(ctg2)

    print("Generating point density (ground focal sum) raster ...")

    pd_ground <- function(chunk, radius, cell_size, cut_ht)
    {
        las <- readLAS(chunk)
        if (is.empty(las)) return(NULL)

        las_1 <- filter_poi(readLAS(chunk), buffer == 0)
        hull <- st_convex_hull(las_1)

        # convert to SpatialPolygons
        bbox <- vect(hull)
        # bbox <- ext(las_1)

        #las <- retrieve_pulses(las)
        density_raster_ground <- rasterize_density(las, res = cell_size, pkg = "terra")[[1]]


        gfw <- focalMat(density_raster_ground, radius, "circle")
        gfw[gfw > 0] = 1
        gfw[gfw == 0] = NA

        Ground_focal <- focal(density_raster_ground, w = gfw, fun = "sum", na.policy = "omit", na.rm = TRUE, fillvalue = NA, expand = FALSE)
        ground_focal <- crop(Ground_focal, bbox)

    }

    opt <- list(need_output_file = TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg2) <- c(0, 0)
    ctg2@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    opt_output_files(ctg2) <- paste0(PD_Ground_folder, "/{*}_PD_Gfocalsum")
    opt_stop_early(ctg2) <- FALSE
    catalog_apply(ctg2, pd_ground, radius = focal_radius, cell_size = cell_size, cut_ht = cut_ht, .options = opt)
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################################################
hh_function <- function(in_las_folder, cell_size, Smooth_by, Min_ws, lawn_range, out_folder, rprocesses) {

    library(lidR)
    library(terra)
    library(sf)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)


    print('Generating Hummock/ Hollow Raster ...')
    ctg <- readLAScatalog(in_las_folder, select = 'xyzc', filter = '-drop_class 7')

    HH_raster <- function(chunk, radius, cell_size, lawn_range, Smooth_by)
    {
        las <- readLAS(chunk)
        if (is.empty(las)) return(NULL)

        las_1 <- filter_poi(readLAS(chunk), buffer == 0)
        hull <- st_convex_hull(las_1)

        # convert to SpatialPolygons
        bbox <- vect(hull)
        # bbox <- ext(las_1)

        #las to DTM
        dtm <- rasterize_terrain(las, res = cell_size, algorithm = tin())


        gfw <- focalMat(dtm, radius, "circle")
        gfw[gfw > 0] = 1
        gfw[gfw == 0] = NA

        rdtm <- focal(dtm, w = gfw, fun = Smooth_by, na.policy = "omit", na.rm = TRUE, fillvalue = NA, expand = TRUE)
        cond_raster <- (rdtm - dtm)
        positive <- abs(lawn_range)
        negative <- positive * -1

        HH <- ifel(cond_raster < negative, 1, ifel(cond_raster > positive, -1, 0))

        cont_hh <- (crop(cond_raster, ext(bbox))) * -1
        hh <- crop(HH, ext(bbox))

        return(list(cont_hh, hh, radius, Smooth_by))
    }

    MultiWrite = function(output_list, file) {
        chh = output_list[[1]]
        hh = output_list[[2]]
        radius = output_list[[3]]
        Smooth_by = output_list[[4]]
        path1 = gsub("@@@_", "CHH_", file)
        path2 = gsub("@@@_", "HH_", file)

        path1 = paste0(path1, "_", Smooth_by, "_", radius, "m.tif")
        path2 = paste0(path2, "_", Smooth_by, "_", radius, "m.tif")

        terra::writeRaster(chh, path1, overwrite = TRUE)
        terra::writeRaster(hh, path2, overwrite = TRUE)

    }

    MultiWriteDriver = list(
      write = MultiWrite,
      extension = "",
      object = "output_list",
      path = "file",
      param = list())

    ctg@output_options$drivers$list <- MultiWriteDriver

    opt_chunk_alignment(ctg) <- c(0, 0)
    opt_output_files(ctg) <- paste0(out_folder, "/CHH_{*}_", Smooth_by, "_", Min_ws, "m")
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    opt_stop_early(ctg) <- TRUE
    out <- catalog_apply(ctg, HH_raster, radius = Min_ws, cell_size = cell_size, lawn_range = lawn_range, Smooth_by = Smooth_by)

    # reset R mutilsession back to default
    plan(sequential)

}

#########################################################################################################################
hh_function_byraster <- function(in_raster, cell_size, Smooth_by, Min_ws, lawn_range, out_folder, rprocesses) {


    library(terra)
    library(utils)
    library(base)
    library(terra)

    print('Generating Hummock/ Hollow Raster ...')
    in_dtm <- rast(in_raster)
    filename <- substr(basename(in_raster), 1, nchar(basename(in_raster)) - 4)

    gfw <- focalMat(in_dtm, Min_ws, "circle")
    gfw[gfw > 0] = 1
    gfw[gfw == 0] = NA

    rdtm <- focal(in_dtm, w = gfw, fun = Smooth_by, na.policy = "omit", na.rm = TRUE, fillvalue = NA, expand = TRUE)
    #   writeRaster(rdtm,paste0(out_folder,"/rdtm_",filename,".tif"),overwrite=TRUE)
    cond_raster <- (rdtm - in_dtm)
    writeRaster(cond_raster, paste0(out_folder, "/CHH_", filename, ".tif"), overwrite = TRUE)
    positive <- abs(lawn_range)
    negative <- positive * -1

    HH <- ifel(cond_raster < negative, 1, ifel(cond_raster > positive, -1, 0))
    writeRaster(HH, paste0(out_folder, "/HH_", filename, ".tif"), overwrite = TRUE)


}


###################################################################################################################################
pd2cellsize <- function(in_las_folder, rprocesses) {

    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)


    print("Calculate raster output's average cell size from point density...")
    if (is(in_las_folder, "LAS") || is(in_las_folder, "LAScatalog"))
    { ctg <- in_las_folder }
    else { ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7') }


    point_density <- sum(ctg@data$Number.of.point.records) / st_area(ctg)
    mean_pd = (3 / point_density)^(1 / 2)
    cell_size = round(0.05 * round(mean_pd / 0.05), 2)
    return(cell_size)
}

##################################################################################

points2trees <- function(in_folder, is_normalized, hmin, out_folder, rprocesses, CHMcell_size, cell_size)
{

    library(lidR)
    library(terra)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #normailize point cloud using K-nearest neighbour IDW
    if (is_normalized) {
        n_las <- readLAScatalog(in_folder, filter = '-drop_class 7 -drop_z_below 0')
    }
    else {
        #read Las file and drop any noise from the point cloud
        ctg <- readLAScatalog(in_folder, filter = '-drop_class 7')
        opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/normalized/n_{*}")
        print("Normalize lidar data...")
        opt_progress(ctg) <- TRUE
        n_las <- normalize_height(ctg, algorithm = knnidw())
        opt_filter(n_las) <- '-drop_class 7 -drop_z_below 0' }

    #     # create a CHM from point cloud for visualization
    if (CHMcell_size != -999) {
        print("Generating normalized CHM ...")
        opt_output_files(n_las) <- opt_output_files(n_las) <- paste0(out_folder, "/chm/{*}_chm")
        n_las@output_options$
          drivers$
          SpatRaster$
          param$
          overwrite <- TRUE
        n_las@output_options$
          drivers$
          Raster$
          param$
          overwrite <- TRUE
        opt_progress(n_las) <- TRUE
        #     chm <- rasterize_canopy(n_las, cell_size, pitfree(thresholds = c(0,3,10,15,22,30,38), max_edge = c(0, 1.5)), pkg = "terra")
        chm <- rasterize_canopy(n_las, CHMcell_size, dsmtin(max_edge = (3 * CHMcell_size)), pkg = "terra") }


    print("Compute approximate tree positions ...")

    ctg_detect_tree <- function(chunk, hmin, out_folder, cell_size) {
        las <- readLAS(chunk)               # read the chunk
        if (is.empty(las)) return(NULL)     # exit if empty
        #         quarter_ht<- ((las@header@PHB$`Max Z` + las@header@PHB$`Min Z`)/4)

        f <- function(x) {
            #         y = 0.4443*(x^0.7874)
            y = 0.478676 * (x^0.695289)  #base on Plot4209, 4207 and 4203
            y[x < hmin] <- 0.478676 * (hmin^0.695289) # Min_ws # smallest window
            #   y[x > (quarter_ht)] <- 0.478676*(quarter_ht^0.695289)    # largest window
            #     y= 0.39328*x
            #     y[x <hmin ] <- 0.39328*hmin # largest window
            #     y[x > (quarter_ht)] <- 0.39328*quarter_ht    # smallest window

            return(y) }

        # dynamic searching window is based on the function of (tree height x 0.3)
        # dynamic window
        ttop <- locate_trees(las, lmf(ws = f, hmin = hmin, shape = "circular"), uniqueness = "gpstime")
        # Fix searching window (Testing only)
        #         ttop <- locate_trees(las, lmf(ws = 3,hmin=hmin,shape="circular"),uniqueness = "gpstime")

        ttop <- crop(vect(ttop), ext(chunk))   # remove the buffer
        # generating number of trees per ha raster
        #    sum_map<-terra::rasterize(ttop,rast(ext(chunk),resolution=cell_size,crs=crs(ttop)),fun=sum)
        #    sum_map<- classify(sum_map, cbind(NA, 0))

        #     return(list(ttop,sum_map))
    }

    options <- list(automerge = TRUE, autocrop = TRUE)
    #    opt_output_files(n_las)<-opt_output_files(n_las)<-paste0(out_folder,"/@@@_{*}")
    opt_output_files(n_las) <- paste0(out_folder, "/{*}_tree_min_", hmin, "_m")
    n_las@output_options$drivers$sf$param$append <- FALSE
    n_las@output_options$
      drivers$
      SpatVector$
      param$
      overwrite <- TRUE
    opt_progress(n_las) <- TRUE
    #    MultiWrite = function(output_list, file){
    #     extent = output_list[[1]]
    #     sum_map = output_list[[2]]
    #     path1 = gsub("@@@_","", file)
    #     path2 = gsub("@@@_","", file)
    #
    #     path1 = paste0(path1, "_trees_above",hmin,"m.shp")
    #     path2 = paste0(path2, "_Trees_counts_above",hmin,"m.tif")
    #
    #     terra::writeVector(extent, path1, overwrite = TRUE)
    #     terra::writeRaster(sum_map,path2,overwrite=TRUE)
    #
    #   }
    #   MultiWriteDriver = list(
    #     write = MultiWrite,
    #     extension = "",
    #     object = "output_list",
    #     path = "file",
    #     param = list())

    #   n_las@output_options$drivers$list <- MultiWriteDriver

    out <- catalog_apply(n_las, ctg_detect_tree, hmin, out_folder, cell_size, .options = options)
    shmin <- as.character(hmin)
    shmin <- gsub("\\.", "p", shmin)
    writeVector(out, paste0(out_folder, "/Merged_ApproxTrees_above_", shmin, "m.shp", overwrite = TRUE))
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################################################################
ht_metrics_lite <- function(in_las_folder, cell_size, out_folder, rprocesses)
{

    library(lidR)
    library(terra)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7 -drop_z_below 0')
    opt_output_files(ctg) <- paste0(out_folder, "/{*}_lite_metrics_z")
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    print('Generating height metrics ...')
    zmetrics_f <- ~list(
      zmax = max(Z),
      zmin = min(Z),
      zsd = sd(Z),
      #       zq25 = quantile(Z, probs = 0.25),
      zq30 = quantile(Z, probs = 0.30),
      #       zq35 = quantile(Z, probs = 0.35),
      zq40 = quantile(Z, probs = 0.40),
      #       zq45 = quantile(Z, probs = 0.45),
      zq50 = quantile(Z, probs = 0.50),
      #       zq55 = quantile(Z, probs = 0.55),
      zq60 = quantile(Z, probs = 0.60),
      #       zq65 = quantile(Z, probs = 0.65),
      zq70 = quantile(Z, probs = 0.70),
      #       zq75 = quantile(Z, probs = 0.75),
      zq80 = quantile(Z, probs = 0.80),
      #       zq85 = quantile(Z, probs = 0.85),
      zq90 = quantile(Z, probs = 0.90),
      #       zq95 = quantile(Z, probs = 0.95),
      zq99 = quantile(Z, probs = 0.99)
    )

    m <- pixel_metrics(ctg, func = zmetrics_f, res = cell_size)
    writeRaster(m, paste0(out_folder, "/Merged_metricsZ.tif"), overwrite = TRUE)

    # reset R mutilsession back to default
    plan(sequential)
}

######################################################################################
veg_cover_percentage <- function(in_las_folder, is_normalized, out_folder, hmin, hmax, cell_size, rprocesses)
{

    library(lidR)
    library(terra)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    if (!(is_normalized)) {
        ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
        opt_output_files(ctg) <- paste0(out_folder, '/normalized/n_{*}')
        opt_progress(ctg) <- TRUE
        print('Normalize point cloud...')
        n_ctg <- normalize_height(ctg, algorithm = knnidw()) }
    else {
        n_ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7 -drop_z_below 0')
    }

    print('Calculating vegetation coverage ...')

    veg_cover_pmetric <- function(chunk, hmin, hmax, out_folder, cell_size)
    {
        las <- readLAS(chunk)

        if (is.empty(las)) return(NULL)

        total_pcount <- pixel_metrics(las, func = ~length(Z), pkg = "terra", res = cell_size, start = c(0, 0))
        # replace NA with 0
        total_pcount <- classify(total_pcount, cbind(NA, 0))
        set.names(total_pcount, "Total_Ncount", index = 1)


        Veg_pcount <- pixel_metrics(las, func = ~length(Z), filter = ~Z >= hmin & Z <= hmax, pkg = "terra", res = cell_size, start = c(0, 0))
        # replace NA with 0
        Veg_pcount <- classify(Veg_pcount, cbind(NA, 0))
        set.names(Veg_pcount, "Veg_Ncount", index = 1)

        veg_percetage <- Veg_pcount / total_pcount
        # replace NA with 0
        veg_percetage <- classify(veg_percetage, cbind(NA, 0))
        set.names(veg_percetage, "Veg_CovPer", index = 1)

        total_pcount <- crop(total_pcount, ext(chunk))
        Veg_pcount <- crop(Veg_pcount, ext(chunk))
        veg_percetage <- crop(veg_percetage, ext(chunk))

        x <- c(total_pcount, Veg_pcount, veg_percetage)

    }

    #
    #         MultiWrite = function(output_list, file)
    #         {
    #           total_pcount = output_list[[1]]
    #           Veg_pcount = output_list[[2]]
    #           veg_CovPer=output_list[[3]]
    #           path1 = gsub("_@@@","_Total_Ncount", file)
    #           path2 = gsub("_@@@","_Veg_Ncount", file)
    #           path3 = gsub("_@@@","_Veg_CovPer", file)
    #           path1 = paste0(path1, ".tif")
    #           path2 = paste0(path2, ".tif")
    #           path3 = paste0(path3, ".tif")
    #
    #           terra::writeRaster(total_pcount,path1,overwrite=TRUE)
    #           terra::writeRaster(Veg_pcount,path2,overwrite=TRUE)
    #           terra::writeRaster(veg_CovPer,path3,overwrite=TRUE)
    #
    #
    #         }
    #         MultiWriteDiver = list(
    #           write = MultiWrite,
    #           extension = "",
    #           object = "output_list",
    #           path = "file",
    #           param = list())

    opt_output_files(n_ctg) <- paste0(out_folder, "/result/{*}_veg_cover_percentage")
    n_ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    #         n_ctg@output_options$drivers$list <- MultiWriteDiver
    out <- catalog_apply(n_ctg, veg_cover_pmetric, hmin, hmax, out_folder, cell_size)

    # reset R mutilsession back to default
    plan(sequential)

}

#########################################################################################
percentage_aboveDBH <- function(in_las_folder, is_normalized, out_folder, DBH, cell_size, rprocesses)
{

    library(lidR)
    library(terra)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)
    sDBH <- DBH
    if (is_normalized) {
        print('Loading normalize point cloud...')
        n_ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7 -drop_z_below 0') }
    else {
        ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
        opt_output_files(ctg) <- paste0(out_folder, '/normalized/n_{*}')
        opt_progress(ctg) <- TRUE
        print('Normalize point cloud...')
        n_ctg <- normalize_height(ctg, algorithm = knnidw())
    }

    print('Calculating percentage returns above DBH ...')

    compute_aboveDBH <- function(chunk, DBH, out_folder, cell_size)
    {
        las <- readLAS(chunk)

        if (is.empty(las)) return(NULL)

        total_pcount <- pixel_metrics(las, func = ~length(NumberOfReturns), pkg = "terra", res = cell_size, start = c(0, 0))

        abvDBH_pcount <- pixel_metrics(las, func = ~length(NumberOfReturns), filter = ~Z >= DBH, pkg = "terra", res = cell_size, start = c(0, 0))

        abvDBH_percetage <- abvDBH_pcount / total_pcount
        set.names(abvDBH_percetage, "Per_abvDBH", index = 1)
        # replace NA with 0
        abvDBH_percetage <- classify(abvDBH_percetage, cbind(NA, 0))
        abvDBH_percetage <- crop(abvDBH_percetage, ext(chunk))
    }

    sDBH <- as.character(sDBH)
    sDBH <- gsub("\\.", "p", sDBH)

    opt_output_files(n_ctg) <- paste0(out_folder, "/{*}_return_above_", sDBH, 'm')
    n_ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    out <- catalog_apply(n_ctg, compute_aboveDBH, DBH, out_folder, cell_size)
    in_file_list = list.files(path = out_folder, pattern = ".tif", all.files = TRUE, full.names = TRUE)
    rast_list <- list()
    for (i in 1:length(in_file_list)) {
        rast_obj <- rast(in_file_list[[i]])
        rast_list <- c(rast_list, rast_obj)
    }
    terra::mosaic(terra::sprc(rast_list), fun = "first", filename = paste0(out_folder, "/Merged__return_above_", sDBH, 'm'), overwrite = TRUE)


    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################
normalized_lidar_knnidw <- function(in_las_folder, out_folder, rprocesses) {

    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/normalized/n_{*}")
    print("Normalize lidar data...")
    opt_progress(ctg) <- TRUE
    n_las <- normalize_height(ctg, algorithm = knnidw())
    # reset R mutilsession back to default
    plan(sequential)
}

##########################################################################
normalized_lidar_tin <- function(in_las_folder, out_folder, rprocesses) {

    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/normalized/n_{*}")
    print("Normalize lidar data...")
    opt_progress(ctg) <- TRUE

    n_las <- normalize_height(ctg, algorithm = tin())
    # reset R mutilsession back to default
    plan(sequential)
}

##########################################################################
normalized_lidar_kriging <- function(in_las_folder, out_folder, rprocesses) {

    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/normalized/n_{*}")
    print("Normalize lidar data...")
    opt_progress(ctg) <- TRUE
    n_las <- normalize_height(ctg, algorithm = kriging())
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################
chm_by_dsmtin <- function(in_las_folder, out_folder, cell_size, is_normalized, rprocesses) {

    library(lidR)
    library(future)
    if (cell_size < 1.0) { rprocesses = rprocesses / 2 }
    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    if (is_normalized) {
        print("Generating CHM using TIN...")
        opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_chm") }
    else {
        print("Generating DSM using TIN...")
        opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_dsm") }

    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    ctg@output_options$drivers$Raster$param$overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    chm <- rasterize_canopy(ctg, cell_size, dsmtin(max_edge = (3 * cell_size)), pkg = "terra")
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################
chm_by_pitfree <- function(in_las_folder, out_folder, cell_size, is_normalized, rprocesses) {

    library(lidR)
    library(future)
    if (cell_size < 1.0) { rprocesses = rprocesses / 2 }
    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7 -drop_overlap')

    if (is_normalized) {
        print("Generate CHM using Pit-free...")
        opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_chm") }
    else {
        print("Generate DSM using Pit-free...")
        opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_dsm") }

    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    ctg@output_options$drivers$Raster$param$overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    chm <- rasterize_canopy(ctg, cell_size, pitfree(thresholds = c(0, 3, 5, 10, 15, 20), max_edge = c(0, 3 * cell_size), subcircle = (cell_size)))
    # reset R mutilsession back to default

    rast_list <- list()
    for (i in 1:length(chm)) {
        rast_obj <- terra::rast(chm[[i]])
        rast_list <- c(rast_list, rast_obj)
    }
    mosaic(sprc(rast_list), fun = "first", filename = paste0(out_folder, "/Merged_CHM.tif"), overwrite = TRUE)

    plan(sequential)

}

#########################################################################################
dtm_by_knnidw <- function(in_las_folder, out_folder, cell_size, rprocesses) {

    library(lidR)
    library(future)
    if (cell_size < 1.0) { rprocesses = rprocesses / 2 }
    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    print("Generate DTM...")
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_DTM")
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    ctg@output_options$drivers$Raster$param$overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    dtm <- rasterize_terrain(ctg, res = cell_size, algorithm = knnidw())
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################
dtm_by_kriging <- function(in_las_folder, out_folder, cell_size, rprocesses) {

    library(lidR)
    library(future)
    if (cell_size < 1.0) { rprocesses = rprocesses / 2 }
    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    print("Generate DTM...")
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_DTM")
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    ctg@output_options$drivers$Raster$param$overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    dtm <- rasterize_terrain(ctg, res = cell_size, algorithm = kriging())
    # reset R mutilsession back to default
    plan(sequential)
}

#########################################################################################
dtm_by_tin <- function(in_las_folder, out_folder, cell_size, rprocesses) {

    library(lidR)
    library(future)
    library(terra)

    if (cell_size < 1.0) { rprocesses = rprocesses / 2 }
    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    print("Generate DTM...")
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_DTM")
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    #         ctg@output_options$drivers$Raster$param$overwrite <- TRUE
    opt_progress(ctg) <- TRUE
    dtm <- rasterize_terrain(ctg, res = cell_size, algorithm = tin())
    # reset R mutilsession back to default
    plan(sequential)
}

###########################################################################################
laz2las <- function(in_las_folder, out_folder, rprocesses) {

    library(lidR)
    library(future)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    mywriteLAS = function(chunk) {
        las <- readLAS(chunk)

        if (is.empty(las)) return(NULL)
        return(las) }

    #read Laz file and drop any noise from the point cloud
    ctg <- readLAScatalog(in_las_folder)
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/las/{*}")
    opt_laz_compression(ctg) <- FALSE
    print("Saving liDAR (laz) data into las...")
    opt_progress(ctg) <- TRUE
    catalog_apply(ctg, mywriteLAS)
    # reset R mutilsession back to default
    plan(sequential)
}

#############################################################
las_info <- function(in_las_folder, rprocesses) {
    library(lidR)
    library(future)


    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)
    print("loading LiDAR Data")
    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    print(paste0("Data format: v", (ctg@data$Version.Major[1]), ".", (ctg@data$Version.Minor[1])))
    print(paste0("Extent: ", min(ctg@data$Min.X), " ", max(ctg@data$Max.X), " ", min(ctg@data$Min.Y), " ", max(ctg@data$Max.Y)))
    print(paste0("Area: ", round(st_area(ctg) / (1000 * 1000), 2), " units²"))
    print(paste0("Total Pts: ", sum(ctg@data$Number.of.point.records)))
    print(paste0("Density: ", round(sum(ctg@data$Number.of.point.records) / st_area(ctg), 0), " pts/units²"))
    print(paste0("Total num. files: ", length(ctg@data$filename)))


}

#######################################################################################################################################
classify_gnd <- function(in_las_folder, out_folder, slope, class_threshold, cloth_resolution, rigidness) {
    library(lidR)
    library(future)
    library(RCSF)

    print("loading LiDAR Data")
    plan(multisession, workers = 4)
    set_lidr_threads(4)

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class_7 -drop_overlap')
    opt_output_files(ctg) <- paste0(out_folder, "/{*}_gnd_classified")
    opt_laz_compression(ctg) <- FALSE
    gnd_csf <- csf(slope, class_threshold = class_threshold, cloth_resolution = cloth_resolution, rigidness = rigidness, iterations = 500, time_step = 0.65)
    print("Classify ground start....")
    c_ctg <- classify_ground(ctg, gnd_csf)
}

#############################################################################################
conduct_raster <- function(in_las_folder, out_folder, cell_size, rprocesses) {

    library(terra)
    library(lidR)
    library(future)
    library(sf)

    plan(multisession, workers = rprocesses)
    set_lidr_threads(rprocesses)

    #normalized LAS with pulse info

    ctg <- readLAScatalog(in_las_folder, filter = '-drop_class 7')
    opt_progress(ctg) <- TRUE

    print("Generating multiple conductivity raster on:")
    print("CHM, Slope, Roughness, ground point density, intensity raster.")
    print("Idea from Correction, update, and enhancement of vectorial forestry line maps using LiDAR data, a pathfinder, and seven metrics, Jean-Romain Roussel, etl 2022.")

    Q_raster <- function(chunk, cell_size)
    {
        las <- readLAS(chunk)
        if (is.empty(las)) return(NULL)

        las_1 <- filter_poi(readLAS(chunk), buffer == 0)
        hull <- st_convex_hull(las_1)
        bbox <- vect(hull)


        #     message('Generate DTM using Triangulation  ...')
        dtm <- rasterize_terrain(las, res = cell_size, algorithm = tin(max_edge = (3 * cell_size)))

        n_las <- normalize_height(las, dtm)

        #message("Generating slope conductivity raster...")
        slope <- terrain(dtm, "slope", 8)
        slope_range = slope@ptr$range_max - slope@ptr$range_min
        Qslope <- ifel(slope <= slope_range * 0.1, 1, ifel(slope > slope_range * 0.75, 0, (1 - ((slope - slope@ptr$range_min) / slope_range))))
        Qslope[is.na(Qslope)] = 0
        Qslope <- terra::crop(Qslope, bbox)


        #     message("Generating roughness conductivity raster...")
        roughness <- terrain(dtm, "roughness")
        roughness_range = roughness@ptr$range_max - roughness@ptr$range_min
        Qrough <- ifel(roughness <= roughness_range * 0.1, 1, ifel(roughness > roughness_range * 0.8, 0, (1 - ((roughness - roughness@ptr$range_min) / roughness_range))))
        Qrough[is.na(Qrough)] = 0
        Qrough <- terra::crop(Qrough, bbox)

        #     message("Generating edge conductivity raster...")
        #sobel filter
        fx = matrix(c(-1, -2, -1, 0, 0, 0, 1, 2, 1), nrow = 3)
        fy = matrix(c(1, 0, -1, 2, 0, -2, 1, 0, -1), nrow = 3)

        dtm_sobelx = focal(dtm, fx, na.policy = "omit")
        dtm_sobely = focal(dtm, fy, na.policy = "omit")

        dtm_sobel = sqrt(dtm_sobelx**2 + dtm_sobely**2)
        dtm_sobel_range = dtm_sobel@ptr$range_max - dtm_sobel@ptr$range_min
        Qedge <- ifel(dtm_sobel <= dtm_sobel_range * 0.15, 1, ifel(dtm_sobel > dtm_sobel_range * 0.85, 0, (1 - ((dtm_sobel - dtm_sobel@ptr$range_min) / dtm_sobel_range))))
        Qedge[is.na(Qedge)] = 0
        Qedge <- terra::crop(Qedge, bbox)

        #     message('Generate CHM...')
        chm <- rasterize_canopy(n_las, cell_size, dsmtin(max_edge = (3 * cell_size)), pkg = "terra")
        chm_range = chm@ptr$range_max - chm@ptr$range_min
        Qchm <- ifel(chm <= chm_range * 0.1, 1, ifel(chm > chm_range * 0.75, 0, (1 - ((chm - chm@ptr$range_min) / chm_range))))
        Qchm[is.na(Qchm)] = 0
        Qchm <- terra::crop(Qchm, bbox)

        #     message("Generating intensity conductivity raster...")
        #     sensor <- track_sensor(las, Roussel2020(pmin=15))
        #     las <- normalize_intensity(las, range_correction(sensor,Rs=1800 ))
        int_max <- pixel_metrics(las, (~max(Intensity)), cell_size) #,filter = ~ReturnNumber == 1L)
        int_min <- pixel_metrics(las, (~min(Intensity)), cell_size) #,filter = ~ReturnNumber == 1L)
        irange_map <- int_max - int_min
        irange_map[is.na(irange_map)] = 0
        iq2 <- global(irange_map, quantile, probs = 0.05, na.rm = TRUE)[[1]]
        int_map_range <- irange_map@ptr$range_max - irange_map@ptr$range_min
        Qint <- ifel(irange_map <= iq2, 1, ifel(irange_map > int_map_range * 0.75, 0, (1 - ((irange_map - irange_map@ptr$range_min) / int_map_range))))
        Qint[is.na(Qint)] = 0
        Qint <- terra::crop(Qint, bbox)

        #     message("Generating ground point density conductivity raster...")
        g = filter_poi(las, Classification == 2L)
        gpd <- rasterize_density(g, res = cell_size, pkg = "terra")
        #     gpd <- pixel_metrics(las, ~list(length(Z)/0.35**2),res=cell_size,filter=~Classification == 2L)
        gq2 <- global(gpd, quantile, probs = 0.02, na.rm = TRUE)[[1]]
        gpd_range = gpd@ptr$range_max - gpd@ptr$range_min
        Qgpd <- ifel(gpd <= gq2, 0, (gpd - gpd@ptr$range_min) / gpd_range)
        Qgpd[is.na(Qgpd)] = 0
        Qgpd <- terra::crop(Qgpd, bbox)


        #     message("Generating low vegetation density conductivity raster...")
        l = filter_poi(n_las, Z >= 1.0 &
          Z <= 3 &
          !(Classification %in% c(LASWATER, LASGROUND, LASBUILDING)))
        lower_density <- rasterize_density(l, res = cell_size, pkg = "terra")
        #     lower_density <- pixel_metrics(n_las, ~list(length(Z)/0.35**2), cell_size,filter=~(Z>= 0.5 & Z<=3))
        lq2 <- global(lower_density, quantile, probs = 0.02, na.rm = TRUE)[[1]]
        lower_range = lower_density@ptr$range_max - lower_density@ptr$range_min
        Qlower <- ifel(lower_density > lq2, 0, 1)
        Qlower[is.na(Qlower)] = 0
        Qlower <- terra::crop(Qlower, bbox)

        #     message("Generating combined conductivity raster...")
        Conduct <- (Qslope * Qlower * Qedge) * (0.25 * Qchm +
          0.25 * Qgpd +
          0.25 * Qrough +
          0.25 * Qint)
        cost <- Conduct * -1 + Conduct@ptr$range_max
        cost[is.na(cost)] = 1

        dtm <- terra::crop(dtm, bbox)
        dtm[is.na(dtm)] = NaN
        chm <- terra::crop(chm, bbox)
        chm[is.na(chm)] = NaN


        lower_canopy <- -ifel(lower_density > lq2, 1, 0)
        lower_canopy <- ifel(lower_canopy == -1, 1, lower_canopy)
        upper_canopy <- ifel(chm > 3, 1, 0)

        lower_canopy <- extend(lower_canopy, ext(bbox))
        upper_canopy <- extend(upper_canopy, ext(bbox))

        canopy <- ifel(upper_canopy == 1, upper_canopy * lower_canopy, upper_canopy + lower_canopy)
        canopy[is.na(canopy)] = 0


        return(list(Qchm, Qslope, Qrough, Qgpd, Qint, Qedge, Qlower, Conduct, cost, dtm, chm, canopy))

    }

    MultiWrite = function(output_list, file) {
        Qchm = output_list[[1]]
        Qslope = output_list[[2]]
        Qrough = output_list[[3]]
        Qgpd = output_list[[4]]
        Qint = output_list[[5]]
        Qedge = output_list[[6]]
        Qlower = output_list[[7]]
        Conductivity = output_list[[8]]
        Cost = output_list[[9]]
        dtm = output_list[[10]]
        chm = output_list[[11]]
        canopy = output_list[[12]]
        path1 = gsub("@@@", "Qchm", file)
        path2 = gsub("@@@", "Qslope", file)
        path3 = gsub("@@@", "Qrough", file)
        path4 = gsub("@@@", "Qgpd", file)
        path5 = gsub("@@@", "Qint", file)
        path6 = gsub("@@@", "Qedge", file)
        path7 = gsub("@@@", "Qlower", file)
        path8 = gsub("@@@", "Conductivity", file)
        path9 = gsub("@@@", "Cost", file)
        path10 = gsub("@@@", "DTM", file)
        path11 = gsub("@@@", "CHM", file)
        path12 = gsub("@@@", "Canopy", file)

        path1 = paste0(path1, ".tif")
        path2 = paste0(path2, ".tif")
        path3 = paste0(path3, ".tif")
        path4 = paste0(path4, ".tif")
        path5 = paste0(path5, ".tif")
        path6 = paste0(path6, ".tif")
        path7 = paste0(path7, ".tif")
        path8 = paste0(path8, ".tif")
        path9 = paste0(path9, ".tif")
        path10 = paste0(path10, ".tif")
        path11 = paste0(path11, ".tif")
        path12 = paste0(path12, ".tif")

        terra::writeRaster(Qchm, path1, overwrite = TRUE)
        terra::writeRaster(Qslope, path2, overwrite = TRUE)
        terra::writeRaster(Qrough, path3, overwrite = TRUE)
        terra::writeRaster(Qgpd, path4, overwrite = TRUE)
        terra::writeRaster(Qint, path5, overwrite = TRUE)
        terra::writeRaster(Qedge, path6, overwrite = TRUE)
        terra::writeRaster(Qlower, path7, overwrite = TRUE)
        terra::writeRaster(Conductivity, path8, overwrite = TRUE)
        terra::writeRaster(Cost, path9, overwrite = TRUE)
        terra::writeRaster(dtm, path10, overwrite = TRUE)
        terra::writeRaster(chm, path11, overwrite = TRUE)
        terra::writeRaster(canopy, path12, overwrite = TRUE)

    }

    MultiWriteDriver = list(
      write = MultiWrite,
      extension = "",
      object = "output_list",
      path = "file",
      param = list())


    ctg@output_options$drivers$list <- MultiWriteDriver
    opt_output_files(ctg) <- opt_output_files(ctg) <- paste0(out_folder, "/{*}_@@@")
    opt_laz_compression(ctg) <- FALSE
    opt_progress(ctg) <- TRUE
    opt <- list(need_output_file = TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg) <- c(0, 0)
    ctg@output_options$
      drivers$
      SpatRaster$
      param$
      overwrite <- TRUE
    opt_stop_early(ctg) <- TRUE
    catalog_apply(ctg, Q_raster, cell_size = cell_size, .options = opt)
    # reset R mutilsession back to default
    plan(sequential)
}

#####################################################################################################
