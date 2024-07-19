#create a 'rlpi_elai' function
library(terra)

rlpi_elai <- function(pdTotal, pdGround, radius, scan_angle, out_lpi, out_elai) {

    total_focal <- rast(pdTotal)
    ground_focal <- rast(pdGround)

    # pdTotal_SpatRaster <- rast(pdTotal)Aug 24
    # pdGround_SpatRaster <- rast(pdGround)
    ground_focal <- extend(ground_focal, ext(total_focal))

    # lpi
    lpi = ground_focal / total_focal
    #lpi
    lpi[is.infinite(lpi)] = NA

    elai = -cos(((scan_angle / 2.0) / 180) * pi) / 0.5 * log(lpi)
    elai[is.infinite(elai)] = NA
    elai[elai == 0 | elai == -0] = 0

    writeRaster(lpi, out_lpi, overwrite = TRUE)
    writeRaster(elai, out_elai, overwrite = TRUE)

}