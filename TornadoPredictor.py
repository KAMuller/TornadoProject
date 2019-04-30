import pandas as pd
import numpy as np
# data processing
# THings to do:
# 1) get avg values form each station per day (for the whole area) 1 vector
# 2) Make a list of each daily weather values
# 3) The Training data is then the past 3 days prior. EXAMPLE: january 4th has the 1st, 2nd and 3rd.
# 4) Class lables in a separate list
# 5) Store The dates of the tornado
# With Pandas, to use data frame returned by read_csv as 2d array use .values[][]
weather_data = pd.read_csv('Weather_Data.csv', header=0)
print(weather_data.head(1))
norm_weather_data = pd.read_csv('NORM_weather.csv', header=0)
print(norm_weather_data.head(1))
#Get the first tornado dates
T_McClain = pd.read_csv('Tornado_McClain_Train.csv', header=None)
print(T_McClain.head(5))
tornado_dates_m = T_McClain.values[3:-1]
print(tornado_dates_m[0])
tornado_dates_m = tornado_dates_m[:, 2:3]
print(tornado_dates_m)
#Get the second tornado dates




#################################################################################
# classifiers










