#create a 'generate_pd' function
generate_pd <- function(ctg,radius_fr_CHM,focal_radius,cell_size,cache_folder,
    cut_ht,PD_Ground_folder,PD_Total_folder,rprocesses){
    library(terra)
    library(lidR)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    opts <- paste0("-drop_class 7")

    print("Processing using R packages.")

    folder <- paste0(cache_folder,"/nlidar/n_{*}" )
    opt_output_files(ctg) <- opt_output_files(ctg) <- folder
    opt_laz_compression(ctg) <-FALSE
    opt_filter(ctg) <- "-drop_class 7"
    opt_chunk_alignment(ctg) <- c(0,0)


    #normalized LAS with pulse info
   print("Indexing LAS Tiles...")
   lidR:::catalog_laxindex(ctg)
   print("Normalize point cloud using K-nearest neighbour IDW....")
   normalize_height(ctg, algorithm=knnidw())

    print("Generate point density (total focal sum) raster....")

    pd_total <- function(chunk,radius,cell_size)
    {
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)

    las_1 <- filter_poi(readLAS(chunk), buffer==0)
    hull <- st_convex_hull(las_1)
    # bbox <- ext(las_1)

    # convert to SpatialPolygons
    bbox <- vect(hull)

    las <- filter_poi(las, Classification != 7L)
    #las <- retrieve_pulses(las)
    density_raster_total <- rasterize_density(las, res=cell_size,pkg="terra")[[1]]

    tfw <- focalMat(density_raster_total, radius, "circle")

    tfw[tfw>0] = 1
    tfw[tfw==0] = NA

    Total_focal <- focal(density_raster_total, w=tfw, fun="sum", na.rm=TRUE,na.policy="omit",fillvalue=NA,expand=FALSE)
    Total_focal <- crop(Total_focal,bbox)
    }

    opt <- list(need_output_file =TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg) <- c(0,0)
    ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_output_files(ctg) <- paste0(PD_Total_folder,"/{*}_PD_Tfocalsum")
    opt_stop_early(ctg) <- FALSE
    catalog_apply(ctg, pd_total,radius=focal_radius,cell_size=cell_size,.options=opt)

    #load normalized LAS for ground point density
    ht<- paste0("-drop_class 7 -drop_z_above ",cut_ht)
    ctg2<- readLAScatalog( paste0(cache_folder,"/nlidar"), filter = ht)
    lidR:::catalog_laxindex(ctg2)


    print("Generate point density (ground focal sum) raster....")
    pd_ground <- function(chunk,radius,cell_size,cut_ht)
    {
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)

    las_1 <- filter_poi(readLAS(chunk), buffer==0)
    hull <- st_convex_hull(las_1)

    # convert to SpatialPolygons
    bbox <- vect(hull)
    # bbox <- ext(las_1)

    #las <- retrieve_pulses(las)
    density_raster_ground <- rasterize_density(las, res=cell_size,pkg="terra")[[1]]


    gfw <- focalMat(density_raster_ground, radius, "circle")
    gfw[gfw>0] = 1
    gfw[gfw==0] = NA

    Ground_focal <- focal(density_raster_ground, w=gfw, fun="sum",na.policy="omit",na.rm=TRUE,fillvalue=NA,expand=FALSE)
    ground_focal <- crop(Ground_focal,bbox)

    }
    opt <- list(need_output_file =TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg2) <- c(0,0)
    ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_output_files(ctg2) <- paste0(PD_Ground_folder,"/{*}_PD_Gfocalsum")
    opt_stop_early(ctg2) <- FALSE
    catalog_apply(ctg2, pd_ground,radius=focal_radius,cell_size=cell_size,cut_ht=cut_ht,.options=opt)
    # reset R mutilsession back to default
    plan("default")
    }