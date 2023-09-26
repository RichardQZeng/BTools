 pd_routine <- function(chunk){
      las <- readLAS(chunk)

      if (is.empty(las)) return(NULL)
      las <- retrieve_pulses(las)
      output <- density(las)[1]
      # output is a list
      return(list(output))
      }