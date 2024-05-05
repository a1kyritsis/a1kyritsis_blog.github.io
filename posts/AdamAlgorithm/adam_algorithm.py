import torch
import time
import statistics
from sklearn.preprocessing import LabelEncoder
import pandas as pd

MAX_ITERS = 1000

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
            self.w = torch.rand((X.shape[1]))
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

class StochasticDescent(LogisticRegression):

    def __init__(self):
        super().__init__()
        self.t = 0

    def stochastic_optimizer(self, X, y, k, alpha, beta):

        if self.w_prev == None:
            self.w_prev = torch.zeros(X.shape[1]) # w_{k - 1}
        
        if self.w_next == None:
            self.w_next = torch.zeros(X.shape[1]) #w_{k + 1}

        n = X.shape[0] # get number of features
        indices = torch.randperm(n)[[torch.arange(k)]] # select k random indices
        X_mini = X[indices, :] # sample points
        y_mini = y[[indices]]
        self.t += 1 # update the time stamp
        self.w_next = self.w - alpha * self.grad(X_mini, y_mini, self.score(X_mini)) + beta * (self.w - self.w_prev) #update
        self.w_prev = self.w #reassign
        self.w = self.w_next

class ADAM(LogisticRegression):

    def __init__(self):
        super().__init__()
    
        self.m_t = None
        self.v_t = None
        self.t = 0
    

    def adam_optimizer(self, X, y, k, alpha, beta_1, beta_2, tau):
       """
       Performs Stochastic Optimization on the objective
       function using the adam algorithm outlined by
       Kingma and Ba, 2015.
       Here:
           k := batch size
           alpha := step size for learning rate
           beta_1, beta_2 := expodential decay rates
       """
       if self.v_t is None:
           self.v_t = torch.zeros(X.shape[1])
       if self.m_t is None:
           self.m_t = torch.zeros(X.shape[1])
       if self.w is None:
           self.w = torch.rand((X.size()[1]))
    
       n = X.shape[0] # get number of features
       indices = torch.randperm(n)[[torch.arange(k)]] # select k random indices
       X_mini = X[indices, :] # sample points
       y_mini = y[[indices]]
       self.t += 1 # update the time stamp
       g_t = self.grad(X_mini, y_mini, self.score(X_mini)) # find the mini batch gradient
       m_t = beta_1 * self.m_t + (1 - beta_1) * g_t # find the first moment
       v_t = beta_2 * self.v_t + (1 - beta_2) * g_t**2 # find the second moment
       m_hat = 1 / (1 - beta_1**self.t) * m_t # error corrected first moment
       v_hat = 1 / (1 - beta_2**self.t) * v_t # error corrected second moment
       self.w = self.w - (alpha * m_hat / (torch.sqrt(v_hat) + tau)) # update weights

class Utility:

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
    
    def train_logistic_with_test(X_train, y_train, X_test, y_test, alpha, beta, epsilon, LR):
        """
        Takes two sets of feature matrix X and corresponding labels y.
        Trains the model on X_train and y_train, and records the loss
        of the weight vector at iteration k of both train and test sets.
        """
        train_loss = []
        test_loss = []
        step = 0

        while(torch.norm(LR.grad(X_train, y_train, LR.score(X_train))) > epsilon):
            LR.gradientDescentOptimizer(X_train, y_train, alpha, beta)
            train_loss.append(LR.loss(X_train, y_train))
            test_loss.append(LR.loss(X_test, y_test))
            step += 1

        return [step, train_loss, test_loss]
    
    def train_adam(X, y, k, alpha, beta_1, beta_2, tau, epsilon, adam):
        """
        Takes a feature matrix X, labels y, parameters, and a adam optimizer.
        Iterates till gradient is epsilon-tolerable.
        Returns number of iterations to converge and loss at each iteration. 
        """
        L = adam.loss(X, y)
        loss = []
        step = 0
        start_time = time.time()

        while(torch.norm(adam.grad(X, y, adam.score(X))) > epsilon):
            adam.adam_optimizer(X, y, k, alpha, beta_1, beta_2, tau)
            L = adam.loss(X, y)
            loss.append(L)
            step += 1

            if step > MAX_ITERS:

                break
        
        end_time = time.time()

        return [step, loss, end_time - start_time]
    
    def train_adam_with_test(X_train, y_train, X_test, y_test, k, alpha, beta_1, beta_2, tau, epsilon, adam):
        """
        Takes two sets of feature matrix X and corresponding labels y.
        Trains the model on X_train and y_train, and records the loss
        of the weight vector at iteration k of both train and test sets.
        """
        train_loss = []
        test_loss = []
        step = 0

        while(torch.norm(adam.grad(X_train, y_train, adam.score(X_train))) > epsilon):
            adam.adam_optimizer(X_train, y_train, k, alpha, beta_1, beta_2, tau)
            train_loss.append(adam.loss(X_train, y_train))
            test_loss.append(adam.loss(X_test, y_test))
            step += 1
        
        return [step, train_loss, test_loss]


    
    def train_stochastic(X, y, k, alpha, beta, epsilon, S):

        """
        Takes a feature matrix X, labels y, parameters, and a adam optimizer.
        Iterates till gradient is epsilon-tolerable.
        Returns number of iterations to converge and loss at each iteration. 
        """
        L = S.loss(X, y)
        loss = []
        step = 0
        start_time = time.time()

        while(torch.norm(S.grad(X, y, S.score(X))) > epsilon):
            S.stochastic_optimizer(X, y, k, alpha, beta)
            L = S.loss(X, y)
            loss.append(L)
            step += 1

            if step > MAX_ITERS:

                break
        
        end_time = time.time()

        return [step, loss, end_time - start_time]

    def adam_batch_test(n, p, gamma_vals, batch_sizes, trials, adam_params):
        """
        Count iterations to convergence for adam algorithm
        for different noise values (gamma) and batch sizes
        adam_params = [alpha, beta_1, beta_2, tau, epsilon, adam]
        """
        res = []
        for  gamma in gamma_vals:
            X, y = Utility.classification_data(n, gamma, p)
            mean_iters = []
            for k in batch_sizes:
                counts = []
                for _ in range(0, trials):
                    adam_params[5].w = None
                    adam_params[5].w_prev = None
                    adam_params[5].w_next = None
                    adam_params[5].m_t = None
                    adam_params[5].v_t = None
                    adam_params[5].t = 0
                    counts.append(Utility.train_adam(X, y, k, adam_params[0], adam_params[1], adam_params[2], adam_params[3], adam_params[4], adam_params[5])[0])
                mean_iters.append(statistics.mean(counts))
            res.append(mean_iters)
        
        return res

    def algo_accuracy(X, y, algo):

        pred = algo.predict(X)
        matching_count = torch.sum(y == pred).item()
        return (matching_count / len(pred))

    def iterations_by_batch_size(X, y, batch_sizes, alpha, beta_1, beta_2, tau, epsilon, adam):
        """
        Test Adam algorithm with different batch sizes.
        Returns lists iterations and time 
        till gradient epsilon-convergence.
        """
        convergence_iters = []
        convergence_time = []

        for k in batch_sizes:
            adam.w = None
            start_time = time.time()
            res = Utility.train_adam(X, y, k, alpha, beta_1, beta_2, tau, epsilon, adam)
            end_time = time.time()
            convergence_iters.append(res[0])
            convergence_time.append(end_time - start_time)

        return [convergence_iters, convergence_time]

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

class Penguins:

    def loadTrainingData():
        #loads training data as pickle object
        try:

            penguins = pd.read_pickle('penguinsDataFrame.pkl')

        except FileNotFoundError:

            print("Bootstrapping...")
            train_url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/palmer-penguins/train.csv"
            penguins = pd.read_csv(train_url)
            penguins.to_pickle('penguinsDataFrame.pkl')

        return penguins

    def prepareData(df):

        le = LabelEncoder()
        df = df.drop(["studyName", "Sample Number", "Individual ID", "Date Egg", "Comments", "Region"], axis = 1)
        le.fit(df["Species"])
        df = df[df["Sex"] != "."]
        df = df.dropna()
        y = le.transform(df["Species"]) #le.fit(train["Species"])
        df = df.drop(["Species"], axis = 1)
        df["Species"] = y #add species back since I would like in the DF
        df = pd.get_dummies(df)
        return df, y


