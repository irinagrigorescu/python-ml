import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from generatePlot import *

# Import Data
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)

# Get labels
#print df.tail(
datasetSize = df.count()[0]
y = df.iloc[0:datasetSize/3*2, 4].values
#y = df.iloc[0:datasetSize, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
#print y

# Get a feature
X = df.iloc[0:datasetSize/3*2, [0,2]].values

# Features and epochs
plt.figure(1)

# Plot features
plt.subplot(211)
plt.scatter(X[:datasetSize/3, 0], 
		X[:datasetSize/3, 1], color='red', marker='o', label='setosa')
plt.scatter(X[datasetSize/3:datasetSize/3*2, 0], 
		X[datasetSize/3:datasetSize/3*2, 1], color='blue', marker='x',
		label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
#plt.show()

# Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.subplot(212)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
    
# Classifier at work
fig = plt.figure(2)
# Plot each step
for i in range(0, 10):
    #plt.figure(2)
    ppn = Perceptron(eta=0.1, n_iter=i)
    ppn.fit(X,y)
    # Generate plot
    plot_decision_regions(X, y, classifier=ppn)
    
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.axis([4,8,0,6])
    plt.legend(loc='upper left')
    plt.draw()
    plt.waitforbuttonpress()
    fig.clear()





