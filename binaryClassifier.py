import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def plotData2D(X, filename=None):
    fig = plt.figure()
    axs = fig.add_subplot(111)

    axs.plot(X[0,:], X[1,:], 'ro', label = 'data')
    if filename == None:
        plt.show()
    plt.close()

def plotClusters(X, k, clusters, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # plot the data
    colors = ["red", "green", "blue", "yellow"]
    for i in range(np.size(X,1)):
        plt.scatter(X[0][i], X[1][i],color = colors[clusters[i]])

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def linearClassifier(X, y): #TODO: lift data
    W = np.linalg.inv(np.dot(X, X.T))
    W = np.dot(np.dot(W, X), y)
    return np.dot(X.T, W), np.sum(np.power(np.dot(X.T, W) - y, 2))/np.size(X,1)

if __name__ == "__main__":
    dt = np.dtype([('PassengerId', np.int), ('Pclass', np.int), ('Name', np.str_, 16), ('Sex', np.str_, 16),
    ('Age', np.float), ('SibSp', np.int), ('Parch', np.int), ('Ticket', np.int),
    ('Fare', np.float), ('Cabin', np.str_, 16), ('Embarked', np.str_, 16)])

    train_data = pd.read_csv('data/train.csv').fillna(0).values
    test_data = pd.read_csv('data/test.csv').values
    # print(data)

    # read information into 1D arrays
    pId = np.copy(train_data[:,0].astype(np.int))
    y = np.copy(train_data[:,1].astype(np.int))
    pClass = np.copy(train_data[:,2].astype(np.int))
    name = np.copy(train_data[:,3].astype(np.str_))
    sex = np.copy(train_data[:,4].astype(np.str_))
    sex = np.where(sex == 'male', 0, 1)
    age = np.copy(train_data[:,5].astype(np.float))
    sibSp = np.copy(train_data[:,6].astype(np.int))
    parch = np.copy(train_data[:,7].astype(np.int))
    ticket = np.copy(train_data[:,8].astype(np.str_))
    fare = np.copy(train_data[:,9].astype(np.int))
    cabin = np.copy(train_data[:,10].astype(np.str_))
    embarked = np.copy(train_data[:,11].astype(np.str_)) #TODO: map to int
    Z = np.vstack((pClass, age))
    # plotData2D(Z)
    # plotClusters(Z, 2, y)
    newY, error = linearClassifier(Z, y)
    print(newY)
    print(error)
