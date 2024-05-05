#Initialize Models
logistic = LogisticRegression()
adam = ADAM()
#Define Testing Parameters and Generate Data
n = 50
p = 100
gamma = .12
epsilon = .01
#logistic parameters
alpha_L = .8
beta_L = .9
#Adam parameters
k_batch = 5
alpha = .5
beta_1 = .9
beta_2 = .999
tau = 1 / 10**8

X_test, y_test = Utility.classification_data(n, gamma, p)
#train logistic to 100 percent
while True:

    X_train, y_train = Utility.classification_data(n, gamma, p)
    res_L = Utility.train_logistic_with_test(X_train, y_train, X_test, y_test, alpha, beta_L, epsilon, logistic)
    pred = logistic.predict(X_train)
    matching_count = torch.sum(y_train == pred).item()
    percentage_match_L = (matching_count / len(pred))

    if ((percentage_match_L >= .75) and not (math.isnan(logistic.loss(X_train, y_train)))):
        break
    else:
        logistic.w = None
        logistic.w_next = None
        logistic.w_prev = None

#Printing Accuracy Results
print("Training Accuracy: " + str(percentage_match_L))
pred = (torch.matmul(X_test, logistic.w) > 0).int()
matching_count = torch.sum(y_test == pred).item()
percentage_match_L = (matching_count / len(pred))
print("Testing Accuracy: " + str(percentage_match_L))

while True:

    X_train, y_train = Utility.classification_data(n, gamma, p)
    res_A = Utility.train_adam_with_test(X_train, y_train, X_test, y_test, k_batch, alpha, beta_1, beta_2, tau, epsilon, adam)
    pred = adam.predict(X_train)
    matching_count = torch.sum(y_train == pred).item()
    percentage_match_A = (matching_count / len(pred))

    if ((percentage_match_A >= .75) and not (math.isnan(adam.loss(X_train, y_train)))):
        break
    else:
        adam.w = None
        adam.w_next = None
        adam.w_prev = None
        adam.t = 0
        adam.m_t = None
        adam.v_t = None

#Printing Accuracy Results
print("Training Accuracy: " + str(percentage_match_A))
pred = (torch.matmul(X_test, adam.w) > 0).int()
matching_count = torch.sum(y_test == pred).item()
percentage_match_A = (matching_count / len(pred))
print("Testing Accuracy: " + str(percentage_match_A))

#Loss over Iterations
[num_iters_L, train_loss_L, test_loss_L] = res_L
[num_iters_A, train_loss_A, test_loss_A]= res_A
iters_L = torch.arange(0, num_iters_L)
iters_A = torch.arange(0, num_iters_A)
plt.plot(iters_L, train_loss_L, color = "blue", label = "Logistic Regression Training Loss")
plt.plot(iters_L, test_loss_L, color = "blue", linestyle = "--", label = "Logistic Regression Testing Loss")
plt.plot(iters_A, train_loss_A, color = "red", label = "ADAM Training Loss")
plt.plot(iters_A, test_loss_A, color = "red", linestyle = "--", label = "ADAM Testing Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Model Loss Over Iterations on Training and Testing Sets")
plt.tight_layout()
plt.legend()