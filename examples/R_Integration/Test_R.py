# -*- coding: utf-8 -*-


import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('processing_R.R')

# Loading the function we have defined in R.
filter_country_function_r = robjects.globalenv['filter_country']

# Reading and processing data
df = pd.read_csv("countries.csv")

#converting it into r object for passing into r function
with (ro.default_converter + pandas2ri.converter).context():
  df_r = ro.conversion.get_conversion().py2rpy(df)
  

#Invoking the R function and getting the result
df_result_r = filter_country_function_r(df_r, 'Canada')

#Converting it back to a pandas dataframe.
with (ro.default_converter + pandas2ri.converter).context():
  df_result = ro.conversion.get_conversion().rpy2py(df_result_r)

