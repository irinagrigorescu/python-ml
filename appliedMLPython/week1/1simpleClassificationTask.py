import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn


################### Data handling
# Read in the file with header:
# fruit_label | fruit_name | fruit_subtype | mass | width | height | color_score
fruits = pd.read_table('fruit_data_with_colors.txt')

# Have a look at the first 5 entries
print "\nFirst 5 entries are: \n {} \n".format(fruits.head())

# Create dictionary with:
#        key:   fruit label (1/2/3/4) and 
#        value: fruit name  (apple/mandarin/orange/lemon)
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print "The label/name pairing is: \n {} \n".format(str(lookup_fruit_name))

# Store the features of the fruits in X
X = fruits[['height', 'width', 'mass', 'color_score']]

# Store the labels of the fruits in y
y = fruits['fruit_label']

# Create training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scatter plot the data where colors mean fruit labels
cmap = cm.get_cmap('gnuplot')
#scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# Plotting a 3D scatter plot mass/width/height
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection = '3d')
ax.scatter(X_train['mass'], X_train['width'], X_train['height'], c = y_train, marker = 'o', s=100, cmap=cmap)
ax.set_xlabel('mass')
ax.set_ylabel('width')
ax.set_zlabel('height')
plt.title('Mass vs Width vs Height of training data fruits')

################### Learning from width/height/color_score
# Use only mass, width and height for training and testing
X = fruits[['width', 'height', 'color_score']]
y = fruits['fruit_label']

# The default split is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create k nearest neighbours classifier object
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit the classifier (train) on the training data
knn.fit(X_train, y_train)

# Calculate the accuracy of the classifier
print "Score of kNN algorithm is: {}".format(knn.score(X_test, y_test))

################### Predict (test)
# Predict a small fruit with mass 20g, width 4.3 cm, height 5.5 cm, and color score 0.79
fruit_prediction1 = knn.predict([[4.3, 5.5, 0.79]]) # fruit label: just an array([label])
print "\nPredict a small fruit with width 4.3 cm, height 5.5 cm, and color score 0.79: "
print "{} - {} \n".format(fruit_prediction1[0], str(lookup_fruit_name[fruit_prediction1[0]]))

# Predict a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm, and color score 0.7
fruit_prediction2 = knn.predict([[6.3, 8.5, 0.7]])
print "\nPredict a larger, elongated fruit with width 6.3 cm, height 8.5 cm, and color score 0.7: "
print "{} - {} \n".format(fruit_prediction2[0], str(lookup_fruit_name[fruit_prediction2[0]]))

# Predict all test data
fruit_prediction_test = knn.predict(X_test) # fruit labels
print "Actual fruit - Predicted fruit"
for i in range(0,y_test.shape[0]):
	#print y_test.iloc[i]
	#print fruit_prediction_test[i]
	print "{} - {}".format(fruits.fruit_name[y_test.iloc[i]], \
						   fruits.fruit_name[fruit_prediction_test[i]])

# Look at two features and where the predicted is
#fig, axarr = plt.subplots(2, 2)
fig2 = plt.figure()
j = 0
colorsPlot = ['red', 'green', 'blue', 'black']
legendPlot = [None] * 4;
for i in y_train.unique():
	#axarr[i%2,j%2].scatter(X_train[y_train == i].mass, X_train[y_train == i].width, c=colorsPlot[i-1], marker='o', s=40)
	#axarr[i%2,j%2].set_xlabel('mass [g]')
	#axarr[i%2,j%2].set_ylabel('width [cm]')	
	#axarr[i%1,j%2].set_title(str(lookup_fruit_name[i]))
	#j = j+1
	#print colorsPlot[i-1]
	plt.scatter(X_train[y_train == i].width, X_train[y_train == i].height, c=colorsPlot[i-1], alpha=0.5, marker='o', s=40)
	legendPlot[j] = lookup_fruit_name[i]
	j=j+1

# Add prediction 1 to the plot
plt.scatter(4.3, 5.5, c=colorsPlot[fruit_prediction1[0]-1], marker='*', s=50)
# Add prediction 2 to the plot
plt.scatter(6.3, 8.5, c=colorsPlot[fruit_prediction2[0]-1], marker='*', s=50)

plt.xlabel('widhth [cm]')
plt.ylabel('height [cm]')
plt.title('Width vs Height for all fruits')
plt.legend(legendPlot, loc=2)

# Plot the decision boundaries of the k-NN classifier
#plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors

################### Look at the algorithm's performance
# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

fig3 = plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);


# How sensitive is k-NN classification accuracy to the train/test split proportion?
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');






# Show plots:
plt.show()


