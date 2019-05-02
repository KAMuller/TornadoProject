from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# data processing
# THings to do:
# 1) get avg values form each station per day (for the whole area) 1 vector
# 2) Make a list of each daily weather values
# 3) The Training data is then the past 3 days prior. EXAMPLE: january 4th has the 1st, 2nd and 3rd.
# 4) Class lables in a separate list
# 5) Store The dates of the tornado





#################################################################################
# classifiers

# oversampling techniques

# random sampling
trainList = []
trainClass = []

n = 2
for x in range(len(trainList)):
    if trainClass[x] == 1:
        for y in range(n-1):
            trainList.append(trainList[x])

def treeClassify(trainX, trainY, testX):
    treeCLF = DecisionTreeClassifier()
    treeCLF.fit(trainX, trainY)
    return treeCLF.predict(testX)


# function for naive bayes classifier, returns predicted values for textX
def bayesClassify(trainX, trainY, testX):
    nbCLF = GaussianNB()
    nbCLF.fit(trainX, trainY)
    return nbCLF.predict(testX)


# function for neural network classifier, returns predicted values for textX
def neuralClassify(trainX, trainY, testX):
    nnCLF = MLPClassifier()
    nnCLF.fit(trainX, trainY)
    return nnCLF.predict(testX)







