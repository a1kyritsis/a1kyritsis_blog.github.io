import torch
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time

class LinearModel:
    #Score and predict
    def __init__(self):
        self.w = None
      

    def score(self, X):
        #compute the score for each data point in the feature matrix X.
        #The formula for ith entry of s is s[i] = <self.w, x[i]>
        #X := feature matrix_{n x p}, n = # points, p = # features
        # RETURN: torch.Tensor, vector of scores
        if self.w is None:
            self.w = torch.rand((X.size()[1]))
        #SCORE
        return torch.matmul(X, self.w) #returns an n x 1 vector (column)
    
    def predict(self, X):
        #positive prediction iff score > 0 . Returns preditions, y_hat
        score_vector = self.score(X)
        #r = torch.where(score_vector >= 0, torch.tensor(1.0), torch.tensor(0.0))
        return (score_vector > 0).int()
    
class LogisticRegression(LinearModel):

    def __init__(self):
        super().__init__()
        self.w_prev = None
        self.w_next = None

    def sigmoid(self, s):
      #vectorized sigmoid function
      return (1 + torch.exp(-1 * s)).pow_(-1)

    
    def loss(self, X, y):
        #assume X are the feautres and y the true values.
        #computes loss
        s = self.score(X) #get scores
        return torch.mean(-1 * y * torch.log(self.sigmoid(s)) - (1 - y)*torch.log(1 - self.sigmoid(s))) #calculate loss
    
    def grad(self, X, y, s):
        #computes the gradient
        n = X.shape[0]
        return (1/n) * torch.sum((X * (self.sigmoid(s) - y).unsqueeze(1)), dim = 0) #calculate gradient (one liner!)

    def gradientDescentOptimizer(self, X, y, alpha, beta):
        #spicy gradient descent
        if self.w_prev == None:
            self.w_prev = torch.zeros(X.shape[1]) # w_{k - 1}
        
        if self.w_next == None:
            self.w_next = torch.zeros(X.shape[1]) #w_{k + 1}
        
        s = self.score(X) #initialize score for w
        self.w_next = self.w - alpha * self.grad(X, y, s) + beta * (self.w - self.w_prev) #update
        self.w_prev = self.w #reassign
        self.w = self.w_next

class NewtonMethod(LogisticRegression):

    def __init__(self):
        super().__init__()

    def hessian(self, X, s):
        #computes the Hessian of Logistic Loss Function
        n = X.shape[0] #get the number of rows
        D = torch.zeros(n, n) #initialize the diagonal matrix
        S = self.sigmoid(s) * (1 - self.sigmoid(s)) #put the scores in the desired form
        D.diagonal().copy_(S) #set the diagonal values
        return X.t() @ D @ X # the hessian is found


    def NewtonOptimizer(self, X, y, alpha):
        s = self.score(X)
        self.w = self.w - alpha * torch.inverse(self.hessian(X, s)) @ self.grad(X, y, s)

class Utility:
    """
    Method used to generate experiment data and visualizations
    """
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

    def train_newton(X, y, alpha, epsilon, NO):
        """"
        takes a feature matrix X, labels y, parameters, and a NewtonOptimizer
        Iterates till loss is epsilon-tolerable.
        returns number of iterations to convergence & loss at each iteration. 
        """
        L = NO.loss(X, y)
        loss = []
        gradients = []
        step = 0
        start_time = time.time()

        while (torch.norm(NO.grad(X, y, NO.score(X))) > epsilon):
            NO.NewtonOptimizer(X, y, alpha)
            L = NO.loss(X, y)
            print(torch.norm(NO.grad(X, y, NO.score(X))))
            loss.append(L)
            step += 1

        end_time = time.time()
        
        return [step, loss, end_time - start_time]

    def train_logistic(X, y, alpha, beta, epsilon, LR):
        """
        Takes a feature matrix X, labels y, parameters, and a LogisticRegressionOptimizer.
        Iterates till loss is epsilon-tolerable.
        Returns number of iterations to converge and loss at each iteration. 
        """
        L = LR.loss(X, y)
        loss = []
        start_time = time.time()
        step = 0

        while(torch.norm(LR.grad(X, y, LR.score(X))) > epsilon):

            LR.gradientDescentOptimizer(X, y, alpha, beta)
            L = LR.loss(X, y)
            loss.append(L)
            step += 1

        end_time = time.time()

        return [step, loss, end_time - start_time]

class Circle:
    """
    Generates non linear data.
    Namely, in a circle!
    """
    def generate_points(radius, n, gamma):
        """
        Generates n pointsin a circle with radius
        and randomness coeffecient gamma 
        """
        points = []
        theta_values = np.linspace(0, 2*np.pi, n)
        for theta in theta_values:
            x = radius * np.cos(theta) + np.random.uniform(-gamma, gamma)
            y = radius * np.sin(theta) + np.random.uniform(-gamma, gamma)
            points.append([x, y])
        return torch.tensor(points)

    def plot_points(points_list):
        """
        Plots set of points from list. Returns
        """
        fig, ax = plt.subplots(figsize=(6,6))

        for points in points_list:

            ax.scatter(points[:,0], points[:, 1])

        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_title('Non-Linear Test Data')
        ax.axis('equal')
        ax.grid(True)
        return fig




    