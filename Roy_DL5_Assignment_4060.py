import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import time
import sklearn
import sklearn.datasets
from DL5 import *
import unit10.utils as u10

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

test_model = DLModel("Test Mini Batch")
X_assess, Y_assess, mini_batch_size = u10.random_mini_batches_test_case()
X_assess = np.array(X_assess)
Y_assess = np.array(Y_assess)
mini_batches = test_model.random_mini_batches(X_assess, Y_assess, mini_batch_size, seed = 0)
print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

train_X, train_Y = u10.load_minibatch_dataset()
plt.show()
layer1 = DLLayer("layer 1", 64, (2,), learning_rate = 0.05, activation = "relu", W_initialization="He")
hidden1 = DLLayer("hidden 1", 32, (64,), learning_rate = 0.05, activation = "relu", W_initialization="He")
hidden2 = DLLayer("hidden 2", 5, (32,), learning_rate = 0.05, activation = "relu", W_initialization="He")
output = DLLayer("output", 1, (5,), learning_rate = 0.05, activation = "sigmoid", W_initialization="He")
model = DLModel()
model.add(layer1)
model.add(hidden1)
model.add(hidden2)
model.add(output)
model.compile("cross entropy")

def run_model(model, num_epocs, minibatch_size):
    tic = time.time()
    costs = model.train(train_X, train_Y, num_epocs, minibatch_size)
    toc = time.time()
    print (f"time (ms): {1000*(toc-tic)}")

    u10.print_costs(costs,num_epocs)
    train_predict = model.forward_propagation(train_X) > 0.7
    accuracy = np.sum(train_predict == train_Y)/train_X.shape[1]
    print("accuracy:", str(accuracy))
    plt.title("Model with no mini batches")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    u10.plot_decision_boundary(model, train_X, train_Y)

run_model(model, 20000,train_X.shape[1])

train_X, train_Y = u10.load_minibatch_dataset()
plt.show()
layer1 = DLLayer("layer 1", 64, (2,), learning_rate = 0.05, activation = "relu", W_initialization="He")
hidden1 = DLLayer("hidden 1", 32, (64,), learning_rate = 0.05, activation = "relu", W_initialization="He")
hidden2 = DLLayer("hidden 2", 5, (32,), learning_rate = 0.05, activation = "relu", W_initialization="He")
output = DLLayer("output", 1, (5,), learning_rate = 0.05, activation = "sigmoid", W_initialization="He")
model = DLModel()
model.add(layer1)
model.add(hidden1)
model.add(hidden2)
model.add(output)
model.compile("cross entropy")
run_model(model, 4000,64)