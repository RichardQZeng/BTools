chm2trees<-function(in_chm,Min_ws,hmin,out_folder,rprocesses)
  {
    update.packages(list('terra','lidR','future'))
    library(lidR)
    library(terra)
    library(future)


    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    current_chm<-rast(in_chm)
    cell_size<-res(current_chm)[1]
    # find the highest point of CHM
    tallest_ht=minmax(current_chm)[2]

#Reforestation Standard of Alberta 2018
#(https://www1.agric.gov.ab.ca/$department/deptdocs.nsf/all/formain15749/$FILE/reforestation-standard-alberta-may1-2018.pdf, p.53)
#Live crown ratio is the proportion of total stem length that is covered by living branches. It is expressed as a percentage or decimal of the total tree height. Live crown ratio is a useful indicator of the status of the tree in relation to vigor, photosynthetic leaf area, and is inversely related to stocking density. It is assumed that live crown ratio must be greater than 0.3 (30%) in order for the tree to release well

      if (Min_ws>= (0.3*hmin)) {
            (Min_ws<-Min_ws)}else{
            (Min_ws<-0.3*hmin)}
       f<-function(x){
          y <- (x*0.3)+Min_ws
          y[x <hmin ] <- (Min_ws) # Smallest Crown
          y[x > tallest_ht] <- (tallest_ht*0.3) # Largest Crown
         return(y)
        }

    out_ttop_filename=paste0(out_folder,"/",substr(basename(in_chm),1,nchar(basename(in_chm))-4),".shp")

    ttop <- locate_trees(current_chm, lmf(ws = f,hmin=hmin,shape="circular"),uniqueness = "bitmerge")

    x<-vect(ttop)
    writeVector(x,out_ttop_filename,overwrite=TRUE)
    #st_write(ttop,out_ttop_filename)

  }

##################################################################################################################
#create a 'generate_pd' function
generate_pd <- function(ctg,radius_fr_CHM,focal_radius,cell_size,cache_folder,
    cut_ht,PD_Ground_folder,PD_Total_folder,rprocesses){
    update.packages(list('terra','lidR','future'))
    library(terra)
    library(lidR)
    library(future)

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
#########################################################################################################################
hh_function <- function(in_las_folder,cell_size, Min_ws, lawn_range, out_folder,rprocesses){
update.packages(list('terra','lidR','future','sf'))
library(lidR)
library(terra)
library(sf)
library(future)

plan(multisession,workers=rprocesses)
set_lidr_threads(rprocesses)


print('Generate Hummock/ Hollow Raster....')
ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')

HH_raster <- function(chunk,radius,cell_size,lawn_range)
{
  las <- readLAS(chunk)
  if (is.empty(las)) return(NULL)

  las_1 <- filter_poi(readLAS(chunk), buffer==0)
  hull <- st_convex_hull(las_1)

  # convert to SpatialPolygons
  bbox <- vect(hull)
  # bbox <- ext(las_1)

  #las to DTM
  dtm<-rasterize_terrain(las,res=cell_size, algorithm = knnidw())


  gfw <- focalMat(dtm, radius, "circle")
  gfw[gfw>0] = 1
  gfw[gfw==0] = NA

  rdtm <- focal(dtm, w=gfw, fun="mean",na.policy="omit",na.rm=TRUE,fillvalue=NA,expand=TRUE)
  cond_raster<- rdtm-dtm
  upper<- abs(lawn_range)
  lower <- upper*-1

  HH<-ifel(cond_raster< lower,-1,ifel(cond_raster > upper,1,0))
  hh<- crop(HH,ext(bbox))
}

opt_chunk_alignment(ctg) <- c(0,0)
opt_output_files(ctg) <- paste0(out_folder,"/result/HH_{*}")
ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
opt_stop_early(ctg) <- TRUE
out<-catalog_apply(ctg,HH_raster,radius=3,cell_size=cell_size,lawn_range=lawn_range)


}
###################################################################################################################################
pd2cellsize <- function(in_las_folder,rprocesses){
    update.packages(list('lidR','future'))
    library(lidR)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)


  print("Calculate average cell size from point density...")
  if (is(in_las_folder,"LAS") & is(in_las_folder, "LAScatalog"))
        {ctg<-in_las_folder}
    else{ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')}

  find_cellsize<-function(chunk){
    las <- readLAS(chunk)

    if (is.empty(las)) return(NULL)
    las <- retrieve_pulses(las)
    output <- density(las)[1]
    # output is a list

  return(list(output))
  }
  opt_progress(ctg) <- TRUE
  pd_list<-catalog_apply(ctg,find_cellsize)
  pulse_density<-mean(unlist(pd_list))
  mean_pd = (3 / pulse_density)^(1 / 2)
  cell_size = round(0.05 * round(mean_pd / 0.05), 2)
  return(cell_size)
}
##################################################################################

points2trees<-function(in_folder,is_normalized,Min_ws,hmin,out_folder,rprocesses,CHMcell_size,cell_size)
  {
  update.packages(list('terra','lidR','future'))
    library(lidR)
    library(terra)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    #normailize point cloud using K-nearest neighbour IDW
    if (!(is_normalized)){
    #read Las file and drop any noise from the point cloud
    ctg<- readLAScatalog(in_folder,filter='-drop_class 7')
    opt_output_files(ctg)<- opt_output_files(ctg)<-paste0(out_folder,"/normalized/n_{*}")
    print("Normalize lidar data...")
    opt_progress(ctg) <- TRUE
    n_las<-normalize_height(ctg,algorithm=knnidw())
    opt_filter(n_las)<-'-drop_below 0'}
    else{
    n_las<- readLAScatalog(in_folder,filter='-drop_class 7 -drop_below 0')
    }

#     # create a CHM from point cloud for visualization
    if (CHMcell_size !=-999){
        print("Generate normalized CHM...")
        opt_output_files(n_las)<- opt_output_files(n_las)<-paste0(out_folder,"/chm/{*}_chm")
        n_las@output_options$drivers$SpatRaster$param$overwrite <- TRUE
        n_las@output_options$drivers$Raster$param$overwrite <- TRUE
        opt_progress(n_las) <- TRUE
    #     chm <- rasterize_canopy(n_las, cell_size, pitfree(thresholds = c(0,3,10,15,22,30,38), max_edge = c(0, 1.5)), pkg = "terra")
        chm <- rasterize_canopy(n_las, CHMcell_size, dsmtin(max_edge = (3*CHMcell_size)), pkg = "terra")}

    print("Compute approximate tree positions ...")
   ctg_detect_tree <- function(chunk,Min_ws,hmin,out_folder,cell_size){
        las <- readLAS(chunk)               # read the chunk
        if (is.empty(las)) return(NULL)     # exit if empty

        # find the highest, average height point of the chunk
        tallest_ht <- las@header@PHB$`Max Z`
        Avg_ht<- ((las@header@PHB$`Max Z` + las@header@PHB$`Min Z`)/2)

   if (Min_ws>= (hmin^(0.3^(1/2)))) {
    (Min_ws<-Min_ws)}else{
      (Min_ws<-(hmin^(0.3^(1/2))))}
  f<-function(x){
    y <-(x^(0.3^(1/2)))
#     y[x <hmin ] <- Min_ws # largest window
#     y[x > (tallest_ht*0.8)] <- (tallest_ht*0.8)^0.3    # smallest window
    y[x <hmin ] <- Min_ws # smallest window
    y[x > (Avg_ht)*1.2] <- (Avg_ht*1.2)^(0.3^(1/2))    # largest window
    return(y)}

# dynamic searching window is based on the function of (tree height x 0.3)
# dynamic window
    ttop <- locate_trees(las, lmf(ws = f,hmin=hmin,shape="circular"),uniqueness = "gpstime")
# Fix searching window (Testing only)
#         ttop <- locate_trees(las, lmf(ws = 1,hmin=hmin,shape="circular"),uniqueness = "gpstime")

   ttop <- crop(vect(ttop), ext(chunk))   # remove the buffer
   sum_map<-terra::rasterize(ttop,rast(ext(chunk),resolution=cell_size,crs=crs(ttop)),fun=sum)
   sum_map<- classify(sum_map, cbind(NA, 0))

    return(list(ttop,sum_map))
   }
   options <- list(automerge = TRUE,autocrop = TRUE)
   opt_output_files(n_las)<-opt_output_files(n_las)<-paste0(out_folder,"/@@@_{*}")
   n_las@output_options$drivers$sf$param$append <- FALSE
   n_las@output_options$drivers$SpatVector$param$overwrite <- TRUE
   opt_progress(n_las) <- TRUE
   MultiWrite = function(output_list, file){
    extent = output_list[[1]]
    sum_map = output_list[[2]]
    path1 = gsub("@@@_","", file)
    path2 = gsub("@@@_","", file)

    path1 = paste0(path1, "_trees_above",hmin,"m.shp")
    path2 = paste0(path2, "_Trees_counts_above",hmin,"m.tif")

    terra::writeVector(extent, path1, overwrite = TRUE)
    terra::writeRaster(sum_map,path2,overwrite=TRUE)

  }
  MultiWriteDriver = list(
    write = MultiWrite,
    extension = "",
    object = "output_list",
    path = "file",
    param = list())

  n_las@output_options$drivers$list <- MultiWriteDriver

   out<-catalog_apply(n_las, ctg_detect_tree,Min_ws,hmin,out_folder,cell_size,.options = options)

  }
#########################################################################################################################################
ht_metrics_lite <- function(in_las_folder,cell_size,out_folder,rprocesses)
{
   update.packages(list('terra','lidR','future'))
    library(lidR)
    library(terra)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')
    opt_output_files(ctg) <- paste0(out_folder,"/{*}_lite_metrics_z")
    ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_progress(ctg) <-  TRUE
    print('Generate height metrics......')
    zmetrics_f <- ~list(
      zmax = max(Z),
      zmin = min(Z),
      zsd = sd(Z),
      zq45 = quantile(Z, probs = 0.45),
      zq50 = quantile(Z, probs = 0.50),
      zq55 = quantile(Z, probs = 0.55),
      zq60 = quantile(Z, probs = 0.60),
      zq65 = quantile(Z, probs = 0.65),
      zq70 = quantile(Z, probs = 0.70),
      zq75 = quantile(Z, probs = 0.75),
      zq80 = quantile(Z, probs = 0.80),
      zq85 = quantile(Z, probs = 0.85),
      zq90 = quantile(Z, probs = 0.90),
      zq95 = quantile(Z, probs = 0.95)
      )

    m<-pixel_metrics(ctg,func=zmetrics_f,res=cell_size)
}

######################################################################################
veg_cover_percentage<-function(in_las_folder,is_normalized,out_folder,hmin,hmax,cell_size,rprocesses)
{
    update.packages(list('terra','lidR','future'))
    library(lidR)
    library(terra)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    if (!(is_normalized)){
    ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')
    opt_output_files(ctg) <- paste0(out_folder,'/normalized/n_{*}')
    opt_progress(ctg) <- TRUE
    print('Normalize point cloud...')
    n_ctg<- normalize_height(ctg,algorithm=knnidw())}
    else{
    n_ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7 -drop_below 0')
    }

        print('Calculating vegetation coverage ....')
        veg_cover_pmetric <- function(chunk,hmin,hmax,out_folder,cell_size)
        {
            las <- readLAS(chunk)

            if (is.empty(las)) return(NULL)

            total_pcount<-pixel_metrics(las,func= ~length(Z),pkg = "terra" ,res=cell_size,start = c(0, 0))
             # replace NA with 0
            total_pcount<- classify(total_pcount, cbind(NA, 0))
            set.names(total_pcount,"Total_Ncount",index=1)


            Veg_pcount<-pixel_metrics(las,func= ~length(Z),filter= ~Z>=hmin & Z<=hmax,pkg = "terra" ,res=cell_size,start = c(0, 0))
             # replace NA with 0
            Veg_pcount<- classify(Veg_pcount, cbind(NA, 0))
            set.names(Veg_pcount,"Veg_Ncount",index=1)

            veg_percetage<- Veg_pcount/total_pcount
             # replace NA with 0
            veg_percetage<- classify(veg_percetage, cbind(NA, 0))
             set.names(veg_percetage,"Veg_CovPer",index=1)

            total_pcount<-crop(total_pcount,ext(chunk))
            Veg_pcount<-crop(Veg_pcount,ext(chunk))
            veg_percetage<-crop(veg_percetage,ext(chunk))

            x<-c(total_pcount,Veg_pcount,veg_percetage)

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

        opt_output_files(n_ctg)<-paste0(out_folder,"/result/{*}_veg_cover_percentage")
        n_ctg@output_options$drivers$SpatRaster$param$overwrite<-TRUE
#         n_ctg@output_options$drivers$list <- MultiWriteDiver
        out<-catalog_apply(n_ctg, veg_cover_pmetric,hmin,hmax,out_folder,cell_size)

}
#########################################################################################
percentage_aboveDBH<-function(in_las_folder,is_normalized,out_folder,DBH,cell_size,rprocesses)
{
    update.packages(list('terra','lidR','future'))
    library(lidR)
    library(terra)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    if (!(is_normalized)){
    ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')
    opt_output_files(ctg) <- paste0(out_folder,'/normalized/n_{*}')
    opt_progress(ctg) <- TRUE
    print('Normalize point cloud...')
    n_ctg<- normalize_height(ctg,algorithm=knnidw())}
    else{
    n_ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7 -drop_below 0')
    }

        print('Calculating percentage returns above DBH ....')
        compute_aboveDBH <- function(chunk,DBH,out_folder,cell_size)
        {
            las <- readLAS(chunk)

            if (is.empty(las)) return(NULL)

            total_pcount<-pixel_metrics(las,func= ~length(NumberOfReturns),pkg = "terra" ,res=cell_size,start = c(0, 0))

            abvDBH_pcount<-pixel_metrics(las,func= ~length(NumberOfReturns),filter= ~Z>=DBH, pkg = "terra" ,res=cell_size,start = c(0, 0))

            abvDBH_percetage<- abvDBH_pcount/total_pcount
            # replace NA with 0
            abvDBH_percetage<- classify(abvDBH_percetage, cbind(NA, 0))
            abvDBH_percetage<-crop(abvDBH_percetage,ext(chunk))


        }

        opt_output_files(n_ctg)<-paste0(out_folder,"/result/{*}_return_above_DBH")
        n_ctg@output_options$drivers$SpatRaster$param$overwrite<-TRUE
        out<-catalog_apply(n_ctg, compute_aboveDBH,DBH,out_folder,cell_size)
}
#########################################################################################
normalized_lidar_knnidw <- function(in_las_folder,out_folder,rprocesses){
    update.packages(list('lidR','future'))
    library(lidR)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

    #read Las file and drop any noise from the point cloud
    ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')
    opt_output_files(ctg)<- opt_output_files(ctg)<-paste0(out_folder,"/normalized/n_{*}")
    print("Normalize lidar data...")
    opt_progress(ctg) <- TRUE
    n_las<-normalize_height(ctg,algorithm=knnidw())
}
#########################################################################################
chm_by_dsmtin <- function(in_las_folder,out_folder,CHMcell_size,rprocesses){
    update.packages(list('lidR','future'))
    library(lidR)
    library(future)

    plan(multisession,workers=rprocesses)
    set_lidr_threads(rprocesses)

        ctg<- readLAScatalog(in_las_folder,filter='-drop_class 7')
        print("Generate CHM...")
        opt_output_files(ctg)<- opt_output_files(ctg)<-paste0(out_folder,"/{*}_chm")
        ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
        ctg@output_options$drivers$Raster$param$overwrite <- TRUE
        opt_progress(ctg) <- TRUE
        chm <- rasterize_canopy(ctg, CHMcell_size, dsmtin(max_edge = (3*CHMcell_size)), pkg = "terra")
    }