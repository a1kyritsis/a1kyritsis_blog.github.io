import torch
import pandas as pd
import math

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
        return (score_vector > 0).int()
    
class LogisticRegression(LinearModel):


    def sigmoid(self, s):
      #vectorized sigmoid function
      return (1 + torch.exp(-1 * s)).pow_(-1)

    
    def loss(self, X, y):
        #assume X are the feautres and y the true values.
        #computes loss
        s = self.score(X) #get scores
        y_ = 2 * y - 1 #convert y to {-1, 1}
        s = self.sigmoid(s) #apply sigmoid function to input score
        return torch.mean(-1 * y_ * torch.log(s) - (1 - y_)*torch.log(1 - s)) #calculate loss
    
    def grad(self, X, y):
        #computes the gradient
        n = X.shape[0]
        s = self.score(X)#get the score
        y_ = 2 * y - 1 #transform labels to {-1 , 1}
        return (1/n) * torch.sum((X * (self.sigmoid(s) - y_).unsqueeze(1)), dim = 0) #calculate gradient (one liner!)

    def gradientDescentOptimizer(self, X, y, alpha, beta, epsilon):
        #spicy gradient descent
        w_prev = torch.zeros(X.shape[1]) # w_{k - 1}
        w_next = torch.zeros(X.shape[1]) #w_{k + 1}
        self.score(X) #initialize score for w
        #TESTING PARAMS
        k = 0
        i = []
        loss = []
        while (torch.norm(self.grad(X, y)) > epsilon): #(self.loss(X, y) > epsilon): #(torch.norm(self.grad(X, y)) > epsilon)
            #TESTING PARAMS
            k = k + 1
            i.append(k)
            #print(self.loss(X, y).item())
            loss.append(self.loss(X, y).item())
            #DESCENT
            w_next = self.w - alpha * self.grad(X, y) + beta * (self.w - w_prev) #update
            w_prev = self.w #reassign
            self.w = w_next

        return [self.w, [i, loss]]

        





    