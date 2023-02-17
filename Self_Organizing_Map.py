# imports
import numpy as np
import pandas as pd
from minisom import MiniSom
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

# read data
data = pd.read_excel('Data.xlsx', header=None)
features = data.values[:, 0:2]
target = data.values[:, 2]

# test train split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# use different colors and markers for each label
markers = ['o', 's', 'D', 'P']
colors = ['C0', 'C1', 'C2', 'C3']
# labels for plot
t = np.zeros(len(target), dtype=int)
t[target == 0] = 0
t[target == 1] = 1
t[target == 2] = 2
t[target == 3] = 3
print("class 0:", (data[data[2] == 0]).count())
print("class 1:", (data[data[2] == 1]).count())
print("class 2:", (data[data[2] == 2]).count())
print("class 3:", (data[data[2] == 3]).count())

for iter in range(4):
    plt.plot((data[data[2] == iter]).values[:, 0], (data[data[2] == iter]).values[:, 1], markers[iter],
             markerfacecolor='None', markeredgecolor=colors[iter], markersize=12, markeredgewidth=2)

plt.show()

# train data can use any size of sigma and lr here:::::
som = MiniSom(10, 10, 2, sigma=2, learning_rate=0.5, activation_distance='euclidean', topology='rectangular',
              neighborhood_function='gaussian')
# get initial weights
initialWeights = som.get_weights()

weightsBeforeTraining = np.reshape(initialWeights, (100, 2))

plt.scatter(weightsBeforeTraining[:, 0], weightsBeforeTraining[:, 1])
plt.title("initial weights")
plt.show()
# train samples
som.train(X_train, 5000, verbose=True)

weightsAfterTraining = np.reshape(initialWeights, (100, 2))
plt.scatter(weightsAfterTraining[:, 0], weightsAfterTraining[:, 1])
plt.title("weights after train")
plt.show()

plt.figure(figsize=(11, 11))

# plot
for cnt, value in enumerate(X_test):
    # print("cnt is :", cnt)  # counter
    # print("xx is: ", value)  # values of winner
    # print("t is :", t[cnt])  # labels
    w = som.winner(value)  # getting the winner
    # place a marker on the winning position for the sample xx
    plt.plot(w[0], w[1], markers[t[cnt]], markerfacecolor='None', markeredgecolor=colors[t[cnt]],
             markersize=12, markeredgewidth=2)
plt.show()

winnerMap = som.labels_map(X_test, y_test)
targetNames = np.unique(y_test)

plt.figure(figsize=(11, 11))
the_grid = GridSpec(11, 11)
# ixx = 0
# ecDist = list()

# get position of the winner weights for plot
for position in winnerMap.keys():
    label_fracs = [winnerMap[position][l] for l in targetNames]
    plt.subplot(the_grid[10 - position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

    # ecDist.append(np.sqrt((X_test[ixx][0] - position[0]) ** 2 + (X_test[ixx][1] - position[1]) ** 2))
    # ixx += 1

# distance of samples from win neron
ecDist = list()
for sample in X_test:
    po = som.winner(sample)
    ecDist.append(np.sqrt((sample[0] - po[0]) ** 2 + (sample[1] - po[1]) ** 2))

print("sum of distances:")
print(sum(ecDist))
# plt.legend(patches,targetNames,bbox_to_anchor=(1,0.5),ncol=4)
plt.show()
# print(ecDist)
