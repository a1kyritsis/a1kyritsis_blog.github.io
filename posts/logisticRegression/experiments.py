import torch
import pandas as pd
import math
import logistic as L
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = "Courier"

def train_logistic(X, y, alpha, beta, epsilon, LR):
    """
    Takes a feature matrix X, labels y, parameters, and a LogisticRegressionOptimizer.
    Iterates till loss is epsilon-tolerable.
    Returns number of iterations to converge and loss at each iteration. 
    """
    L = LR.loss(X, y)
    loss = []
    step = 0

    while(torch.norm(LR.grad(X, y, LR.score(X))) > epsilon):
        LR.gradientDescentOptimizer(X, y, alpha, beta)
        L = LR.loss(X, y)
        loss.append(L)
        step += 1

    return [step, loss]

def classification_data(n_points, noise, p_dims):
    #generates test data for LR model
    #To run: classification_data(n_points = 300, noise = 0.2, p_dims = 2):
    y = torch.arange(n_points) >= int(n_points/2)
    y = 1.0*y
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,p_dims))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)
    
    return X, y

def draw_line(w, x_min, x_max, ax, **kwargs):
    w_ = w.flatten()
    x = torch.linspace(x_min, x_max, 101)
    y = -1 * w[0]/w[1] * x + .75
    l = ax.plot(x, y, **kwargs)


#PART I
#initialize model and test data
"""
n = 100
p = 2
gamma = .12
X, y = classification_data(n, gamma, p)
LR = L.LogisticRegression()
LR.w = None
alpha = .5
beta = .25
epsilon = .01

res = train_logistic(X, y, alpha, beta, epsilon, LR)
w = LR.w
#plotting decision boundary
plt.scatter(X[:,0], X[:, 1], c = y, cmap = "coolwarm")
draw_line(w, -2, 2.5, plt, color = "red", linestyle = "dashed", label = "Decision Boundary")
plt.legend()
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Test Points")
plt.show()
#plotting loss over iterations
[num_iters, loss] = res
print(loss)
i = torch.arange(0, num_iters)
print(loss) 
plt.plot(i, loss, color = "red")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Model Loss Over Iterations")
plt.show()

#PART II
#calling Vanilla LR
n = 100
p = 10
gamma = .25
alpha = .1
beta = 0
epsilon = .01
X, y = classification_data(n, gamma, p)
LR = L.LogisticRegression()
res = train_logistic(X, y, alpha, beta, epsilon, LR)
#Reset and call Spicy LR
LR.w = None
beta = .75
print("calling res momentum")
res_momentum = train_logistic(X, y, alpha, beta, epsilon, LR)
#plotting
[num_iters, loss] = res
[num_iters_momentum, loss_momentum] = res_momentum
iters = torch.arange(0, num_iters)
iters_momentum = torch.arange(0, num_iters_momentum)
plt.plot(iters, loss, color = "red", label = "Vanilla")
plt.plot(iters_momentum, loss_momentum, color = "Blue", label = "Momentum")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Model Loss Over Iterations")
plt.legend()
plt.show()

"""
#PART III
#initialize parameters
beta = .25
n = 50
p = 100
gamma = .5
alpha = .1
beta = .5
epsilon = .01
LR = L.LogisticRegression()
#generate training and testing data
X_test, y_test = classification_data(n, gamma, p)
while True:
    X_train, y_train = classification_data(n, gamma, p)
    res = train_logistic(X_train, y_train, alpha, beta, epsilon, LR)
    pred = LR.predict(X_train)
    matching_count = torch.sum(y_train == pred).item()
    percentage_match = (matching_count / len(pred))
    if (percentage_match == 1):
        break
    else:
        LR.w = None

print("Training Accuracy: " + str(percentage_match))
pred = (torch.matmul(X_test, LR.w) > 0).int()
matching_count = torch.sum(y_test == pred).item()
percentage_match = (matching_count / len(pred))
print("Testing Accuracy: " + str(percentage_match))
