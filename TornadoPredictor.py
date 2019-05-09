# CS 484 Final Project
# by Kalman Muller and Jonathan Barker
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import MiniBatchKMeans
from sklearn. ensemble import RandomForestClassifier

# data processing
# Things to do:
# 1) get avg values form each station per day (for the whole area) 1 vector
# 2) Make a list of each daily weather values
# 3) The Training data is then the past 3 days prior. EXAMPLE: january 4th has the 1st, 2nd and 3rd.
# 4) Class lables in a separate list
# 5) Store The dates of the tornado
# 6) NEED TO MAKE A TEST AND TRAIN SET
# With Pandas, to use data frame returned by read_csv as 2d array use .values[][]
weather_data = pd.read_csv('Weather_Data.csv', header=0, dtype={'STID': str})
# print(weather_data.head(1))
norm_weather_data = pd.read_csv('NORM_weather.csv', header=0, dtype={'STID': str})
# Get the first tornado dates
t_McClain = pd.read_csv('Tornado_McClain_Train.csv', header=None)
tornado_dates_m = t_McClain.values[3:len(t_McClain.values), 2:3]

# Get the second tornado dates
t_Grady = pd.read_csv('Tornado_Grady_Train.csv', header=None)
tornado_dates_g = t_Grady.values[3:len(t_Grady.values), 2:3]

# Get the third tornado dates
t_Cleve = pd.read_csv('Tornado_Cleveland_Train.csv', header=None)
tornado_dates_c = t_Cleve.values[3: len(t_Cleve.values), 2:3]

# TO do list: 3 lists, make the day predictor
# train_list, vectors, remove the test subset
# train_class for each vector (yes or no)
# test_list a subset of a list of vector's 1 year vectors
# combining tornado dates

def nDayVects(n, vectorList, labels):
    nDaysVectors = []
    nDaysClass = []
    hold = []
    for x in range(len(vectorList) - n):
        i = 1
        hold = vectorList[x + n - i]
        while i < n:
            i += 1
            hold = np.concatenate([hold, vectorList[x + n - i]])
        nDaysVectors.append(hold)
        nDaysClass.append(labels[ x + n])
    return nDaysVectors, nDaysClass

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
test_dates = []
test_list = []
weather_vectors = []
weather_dates = []
weather = weather_data.values.tolist()
norm_weather = norm_weather_data.values.tolist()
print(len(weather))
for x in weather:
    year = x[0]
    month = x[1]
    day = x[2]
    date = str(month) + '/' + str(day) + '/' + str(year)
    if (date not in weather_dates) and date not in test_dates:
        # if year == 2015:
        #     test_dates.append(date)
        weather_dates.append(date)
    # if year == 2015:
    #     test_list.append(x[4:len(x)])
    weather_vectors.append(x[4:len(x)])
print(len(weather_vectors))

# get the class labels
labels = []
for x in range(len(weather_dates)):
    if weather_dates[x] in tornadoes:
        labels.append(1)
    else:
        labels.append(0)
print("labels length is: ", len(labels))

# cleaning weather vectors
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
print(weather_vectors[1])
# cleaning up norm data
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
#combining Norm data with the rest of the weather data
count = 0
for x in range(len(weather_vectors)):
    if weather_vectors[x] == 0:
        newline = norm_weather[count]
        weather_vectors[x] = newline[4:len(newline)]
        count += 1
print("The complete weather is: ", weather_vectors[:3])
# Combining weather vectors
weather_train = []
hold = []
i = 0
while i < len(weather_vectors):
    v1 = weather_vectors[i]
    v2 = weather_vectors[i+1]
    v3 = weather_vectors[i+2]
    hold = [v1, v2, v3]
    for x in range(len(hold[0])):
        if hold[0][x] == -1:
            if hold[1][x] != -1:
                hold[0][x] = hold[1][x]
            elif hold[2][x] != -1:
                hold[0][x] = hold[2][x]
        if hold[1][x] == -1:
            if hold[0][x] != -1:
                hold[1][x] = hold[0][x]
            elif hold[2][x] != -1:
                hold[1][x] = hold[2][x]
        if hold[2][x] == -1:
            if hold[1][x] != -1:
                hold[2][x] = hold[1][x]
            elif hold[0][x] != -1:
                hold[2][x] = hold[0][x]
    new_vector = np.mean(hold, axis=0)
    new_vector = np.round(new_vector, decimals=4)
    weather_train.append(new_vector)
    i += 3
weather_train = np.array(weather_train)
print("Average Weather is: ", weather_train[0])

weather_test = []
hold = []
i = 0
while i < len(test_list):
    v1 = test_list[i]
    v2 = test_list[i+1]
    v3 = test_list[i+2]
    hold = [v1, v2, v3]
    for x in range(len(hold[0])):
        if hold[0][x] == -1:
            if hold[1][x] != -1:
                hold[0][x] = hold[1][x]
            elif hold[2][x] != -1:
                hold[0][x] = hold[2][x]
        if hold[1][x] == -1:
            if hold[0][x] != -1:
                hold[1][x] = hold[0][x]
            elif hold[2][x] != -1:
                hold[1][x] = hold[2][x]
        if hold[2][x] == -1:
            if hold[1][x] != -1:
                hold[2][x] = hold[1][x]
            elif hold[0][x] != -1:
                hold[2][x] = hold[0][x]
    new_vector = np.mean(hold, axis=0)
    new_vector = np.round(new_vector, decimals=4)
    weather_test.append(new_vector)
    i += 3
weather_test = np.array(weather_test)
weather_vectors = weather_train
# print(weather_train[:2])
print("There are: ", len(weather_vectors), " training vectors.")
print("There are: ", len(weather_vectors[0]), "parameters.")
print(len(weather_train) + len(weather_test))
# Assign Labels
print("There are: ", len(weather_dates) + len(test_dates), "dates.")

################################
n = 3
################################

# call the function for n days
nVector_data, class_list = nDayVects(n, weather_vectors, labels)
print("train data is length: ", len(nVector_data))

weather_dates = weather_dates[n:len(weather_dates)]

trainList = []
trainClass = []
testList = []
testClass = []

dateCheck = False
for x in range(len(nVector_data)):
    if "2015" in weather_dates[x]:
        dateCheck = True
    if "2016" in weather_dates[x]:
        dateCheck = False
    if dateCheck:
        testList.append(nVector_data[x])
        testClass.append(labels[x])
    else:
        trainList.append(nVector_data[x])
        trainClass.append(labels[x])
trainList = np.array(trainList)
testList = np.array(testList)
print(np.shape(trainList))
print(np.shape(testList))
#################################################################################
# code used to create the sample weights used in calculating the F1 scores
sampleWeights = []
for line in testClass:
    if line == 0:
        sampleWeights.append(1)
    else:
        sampleWeights.append(70)


# duplicating values oversampling function
def multOversamp(trainL, trainC, n):
    global trainList
    for x in range(len(trainL)):
        if trainC[x] == 1:
            for y in range(n-1):
                # print(trainList)
                trainList = np.append(trainList, [trainList[x]], axis=0)
                trainClass.append(1)


# random oversampling with replacement function
def randOversamp(trainL, trainC, frac):
    global trainList
    np.random.seed(100)
    trueClassList = []
    falseCalssList = []
    for x in range(len(trainC)):
        if trainC[x] == 1:
            trueClassList.append(trainL[x])
        else:
            falseCalssList.append(trainL[x])
    trueLen = len(trueClassList)
    tListLen = len(trueClassList)
    currFrac = tListLen/len(falseCalssList)
    while currFrac <= frac:
        index = np.random.randint(0, trueLen)
        trainList = np.append(trainList, [trueClassList[index]], axis=0)
        trainClass.append(1)
        tListLen += 1
        currFrac = tListLen / len(falseCalssList)


# k means undersampling function
def kMeansUndersamp(trainL, trainC, frac):
    global trainList
    trueClassList = []
    falseClassList = []
    for x in range(len(trainC)):
        if trainC[x] == 1:
            trueClassList.append(trainL[x])
        else:
            falseClassList.append(trainL[x])
    tListLen = len(trueClassList)
    currFrac = tListLen / len(falseClassList)
    while currFrac <= frac:
        cluster = MiniBatchKMeans(n_clusters=int(len(falseClassList)/2), init='k-means++', max_iter=50).fit(falseClassList)
        falseClassList = np.floor(cluster.cluster_centers_).astype(np.int)
        currFrac = tListLen / len(falseClassList)
    tTrainList = []
    trainClass.clear()
    for x in range(len(falseClassList)):
        tTrainList.append(falseClassList[x])
        trainClass.append(0)
    for x in range(len(trueClassList)):
        tTrainList.append(trueClassList[x])
        trainClass.append(1)
    trainList = np.array(tTrainList).astype(np.int)


# function for the decision tree classifier, returns predicted values for textX
def treeClassify(trainX, trainY, testX):
    treeCLF = DecisionTreeClassifier(max_depth=5, max_features=17, random_state=97)
    treeCLF.fit(trainX, trainY)
    return treeCLF.predict(testX), treeCLF.predict_proba(testX)


# function for naive bayes classifier, returns predicted values for textX
def bayesClassify(trainX, trainY, testX):
    nbCLF = ComplementNB(alpha=44.5, norm=False)
    nbCLF.fit(np.absolute(trainX), trainY)
    return nbCLF.predict(testX), nbCLF.predict_proba(testX)


# function for neural network classifier, returns predicted values for textX
def neuralClassify(trainX, trainY, testX):
    nnCLF = MLPClassifier(learning_rate='adaptive', random_state=29)
    nnCLF.fit(trainX, trainY)
    return nnCLF.predict(testX), nnCLF.predict_proba(testX)


# function for random forest classifier, returns predicted values for textX
def forestClassify(trainX, trainY, testX):
    forestCLF = RandomForestClassifier(n_estimators=5)
    forestCLF.fit(trainX, trainY)
    return forestCLF.predict(testX), forestCLF.predict_proba(testX)

# function that prints the correctly and incorrectly classified classes from the predicted classes
def scoreVal(predClass, testClass):
    torTrueC = 0
    torTrueW = 0
    torFalseC = 0
    torFalseW = 0
    for x in range(len(predClass)):
        if predClass[x] == 1 and testClass[x] == 1:
            torTrueC += 1
        elif predClass[x] == 1 and testClass[x] == 0:
            torTrueW += 1
        elif predClass[x] == 0 and testClass[x] == 0:
            torFalseC += 1
        elif predClass[x] == 0 and testClass[x] == 1:
            torFalseW += 1
    print("torTrue correct: ", torTrueC)
    print("torTrue wrong: ", torTrueW)
    print("torFalse correct: ", torFalseC)
    print("torFalse wrong: ", torFalseW)

# code used to chose the parameter values for the classifiers

# params = []
# for x in range(2000):
#     params.append(x/2)
# paramScores = []
# for x, param in enumerate(params):
#     tuneclassifer = ComplementNB(alpha=param)
#     tuneclassifer.fit(np.absolute(trainList), trainClass)
#     out = tuneclassifer.predict(testList)
#     paramScores.append(f1_score(testClass, out, average='macro', sample_weight=sampleWeights))
#     print(x)
# bestScore = 0
# for x, line in enumerate(paramScores):
#     if line > bestScore:
#         bestScore = line
#         print(line, x)

# finds the predicted classes from each classifier and prints their f1 score and scoreVal()
print("forest")
forestTestClass, forestTestClassProb = forestClassify(trainList, trainClass, testList)
forestF1 = f1_score(testClass, forestTestClass, average='macro', sample_weight=sampleWeights)
print(forestF1)
scoreVal(forestTestClass, testClass)
print("tree")
treeTestClass, treeTestClassProb = treeClassify(trainList, trainClass, testList)
treeF1 = f1_score(testClass, treeTestClass, average='macro', sample_weight=sampleWeights)
print(treeF1)
scoreVal(treeTestClass, testClass)
print("bayes")
bayesTestClass, bayesTestClassProb = bayesClassify(trainList, trainClass, testList)
bayesF1 = f1_score(testClass, bayesTestClass, average='macro', sample_weight=sampleWeights)
print(bayesF1)
scoreVal(bayesTestClass, testClass)
print("Predictron")
randOversamp(trainList, trainClass, .5)
nnTestClass, nnTestClassProb = neuralClassify(trainList, trainClass, testList)
nnF1 = f1_score(testClass, nnTestClass, average='macro', sample_weight=sampleWeights)
print(nnF1)
scoreVal(nnTestClass, testClass)

# determines the most accurate predicted classes based on the f1 score
if forestF1 > treeF1 and forestF1 > bayesF1 and forestF1 > nnF1:
    predClass = forestTestClass
    predClassProb = forestTestClass
elif treeF1 > bayesF1 and treeF1 > nnF1:
    predClass = treeTestClass
    predClassProb = treeTestClassProb
elif bayesF1 > nnF1:
    predClass = bayesTestClass
    predClassProb = bayesTestClassProb
else:
    predClass = nnTestClass
    predClassProb = nnTestClassProb

print("final prediction classes")
print(predClass)
