import pandas as pd
import sys
import numpy as np
# data processing
# THings to do:
# 1) get avg values form each station per day (for the whole area) 1 vector
# 2) Make a list of each daily weather values
# 3) The Training data is then the past 3 days prior. EXAMPLE: january 4th has the 1st, 2nd and 3rd.
# 4) Class lables in a separate list
# 5) Store The dates of the tornado
# With Pandas, to use data frame returned by read_csv as 2d array use .values[][]
weather_data = pd.read_csv('Weather_Data.csv', header=0, dtype={'STID': str})
# print(weather_data.head(1))
norm_weather_data = pd.read_csv('NORM_weather.csv', header=0, dtype={'STID': str})
# Get the first tornado dates
t_McClain = pd.read_csv('Tornado_McClain_Train.csv', header=None)
tornado_dates_m = t_McClain.values[3:-1, 2:3]

# Get the second tornado dates
t_Grady = pd.read_csv('Tornado_Grady_Train.csv', header=None)
tornado_dates_g = t_Grady.values[3:-1, 2:3]

# Get the third tornado dates
t_Cleve = pd.read_csv('Tornado_Cleveland_Train.csv', header=None)
tornado_dates_c = t_Cleve.values[3:-1, 2:3]


# combining tornado dates
tornadoes = []
for x in tornado_dates_m:
    if x not in tornadoes:
        tornadoes.append(x)
for x in tornado_dates_g:
    if x not in tornadoes:
        tornadoes.append(x)
for x in tornado_dates_c:
    if x not in tornadoes:
        tornadoes.append(x)
tornadoes = np.array(tornadoes)
print("There are: ", len(tornadoes), "tornadoes.")

# Combining Weather data
weather_vectors = []
weather_dates = []
weather = weather_data.values.tolist()
norm_weather = norm_weather_data.values.tolist()
for x in weather:
    year = x[0]
    month = x[1]
    day = x[2]
    date = str(month) + '/' + str(day) + '/' + str(year)
    if date not in weather_dates:
        weather_dates.append(date)
    weather_vectors.append(x[4:-1])

nan_count = 0
i = 0
for x in weather_vectors:
    i += 1
    nan_count = 0
    for y in range(len(weather_vectors[0])):
        if x[y] < -1:
            nan_count += 1
            x[y] = -1
    if nan_count == len(x):
        weather_vectors[i-1] = 0
# print(weather_vectors[:5])
i = 0
for x in norm_weather:
    i += 1
    nan_count = 0
    for y in range(len(norm_weather[0])):
        if type(x[y]) != str:
            if x[y] < -1:
                nan_count += 1
                x[y] = -1
    if nan_count == len(x):
        norm_weather[i-1] = 0
# print("Norm Weather is: ", norm_weather[:5], '\n')
count = 0
for x in range(len(weather_vectors)):
    if weather_vectors[x] == 0:
        newline = norm_weather[count]
        weather_vectors[x] = newline[4:-1]
        count += 1
# print("The complete weather is: ", weather_vectors[:5])
# Combining weather vectors
weather_train = []
hold = []
i = 0
while i < len(weather_vectors):
    v1 = weather_vectors[i]
    v2 = weather_vectors[i+1]
    v3 = weather_vectors[i+2]
    hold = [v1, v2, v3]
    new_vector = np.mean(hold, axis=0)
    new_vector = np.round(new_vector, decimals=4)
    weather_train.append(new_vector)
    i += 3
weather_train = np.array(weather_train)
# print(weather_train[:2])
print("There are: ", len(weather_train), "vectors.")

# Assign Labels
print("There are: ", len(weather_dates), "dates.")
weather_train_labels = []
hold = []
for x in range(len(weather_dates)):
    hold = []
    if weather_dates[x] in tornadoes:
        hold.append(weather_train[x])
        hold.append('Yes')
        weather_train_labels.append(hold)
    else:
        hold.append(weather_train[x])
        hold.append('No')
        weather_train_labels.append(hold)
tornado_count = 0
for x in weather_train_labels:
    if x[len(weather_train_labels[0])-1] == 'Yes':
        tornado_count += 1
print("There are ", tornado_count, "recorded tornadoes!")






#################################################################################
# classifiers










