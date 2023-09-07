
filter_country <- function(df, country){
  #' Preprocessing df to filter country
  #'
  #' This function returns a subset of the df
  #' if the value of the country column contains 
  #' the country we are passing
  #'
  #' @param df The dataframe containing the data 
  #' @param country The country we want to filter
  #
  df = subset(df, df$Country == country)
  return(df)
}