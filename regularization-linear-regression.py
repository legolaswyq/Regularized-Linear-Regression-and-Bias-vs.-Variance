import numpy as np
import scipy.io
import scipy.optimize as opt
import matplotlib.pyplot as plt

# predict the amount of water flowing out of a dam using the change of water level in a reservoir

filename = "ex5data1.mat"
data = scipy.io.loadmat(filename)

# change in water level
X = data["X"]
# water flowing out of the dam
y = data["y"].flatten()
X_test = data["Xtest"]
y_test = data["ytest"].flatten()
X_val = data["Xval"]
y_val = data["yval"].flatten()


def linear_regression(theta, X, y, lamb):
    # X(12,1+1)  theta(2,1) y(12,1)
    m = X.shape[0]
    ones = np.ones([m, 1])
    X = np.hstack([ones, X])
    h = X.dot(theta)
    # cost function
    J = 1 / 2 / m * np.sum(np.power(h - y, 2)) + lamb / 2 / m * np.sum(np.power(theta[1:], 2))
    # gradient   X(12,2)  X.T(2,12) (h-y)(12,1)   sum_error(2,1)\
    sum_error = 1 / m * X.T.dot(h - y)
    # remember to use theta.copy() otherwise it will change the theta value
    temp = theta.copy()
    temp[0] = 0
    gradient = sum_error + lamb / m * temp
    return J, gradient


def f(theta, X, y, lamb):
    J, gradient = linear_regression(theta, X, y, lamb)
    return J


def fprime(theta, X, y, lamb):
    J, gradient = linear_regression(theta, X, y, lamb)
    return gradient


def train_model(X, y, lamb=0):
    feature = X.shape[1] + 1
    init_theta = np.ones(feature)
    train_theta = opt.fmin_cg(f, init_theta, fprime=fprime, args=(X, y, lamb),disp=False,maxiter=200)
    return train_theta


# theta = train_model(X, y)
# print(theta)


# learning curve
def plot_learning_curve(X, y, X_val, y_val, lamb=0):
    m = X.shape[0]
    err_train_repeat = np.zeros([50, m])
    err_val_repeat = np.zeros([50, m])

    for i in range(50):
        for j in range(m):
            # from range(m) choice j random number
            rand_idx = np.random.choice(m, size=j + 1, replace=False)
            theta = train_model(X[rand_idx], y[rand_idx], lamb)
            err_train_repeat[i, j] = f(theta, X[rand_idx], y[rand_idx], 0)
            err_val_repeat[i, j] = f(theta, X_val, y_val, 0)

    err_train = err_train_repeat.sum(axis=0) / 50
    err_val = err_val_repeat.sum(axis=0) / 50

    plt.plot(np.arange(1, m + 1), err_train)
    plt.plot(np.arange(1, m + 1), err_val)
    plt.show()


# plot_learning_curve(X,y,X_val,y_val,1)

def poly_feature(X, P):
    poly_feature = np.zeros([X.shape[0], P])
    for i in range(P):
        poly_feature[:, i] = np.power(X[:, 0], i + 1)
    return poly_feature


def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    norm_feature = (X - mean) / std
    return norm_feature


def ploy_norm_feature(X):
    P = 8
    features = poly_feature(X, P)
    norm_features = feature_normalize(features)
    return norm_features


def plot_data_and_decision_boundary(X, y, lamb=1):
    # plot data set
    plt.scatter(X, y, marker="x", c="red")
    plt.xlabel("change in water level")
    plt.ylabel("water flowing out of the dam")
    plt.title("water level")
    # plot boundary
    # train model using X
    # use linspace to create a continous value for x
    poly_X_for_train = ploy_norm_feature(X)
    theta = train_model(poly_X_for_train, y, lamb)
    x = np.linspace(min(X) - 15, max(X) + 15, 50)
    poly_X = ploy_norm_feature(x)
    ones = np.ones([poly_X.shape[0], 1])
    X_label = np.hstack([ones, poly_X])
    plt.plot(x, X_label.dot(theta))
    plt.show()


# plot_learning_curve(poly_X,y,poly_X_val,y_val,0)
# plot_data_and_decision_boundary(X, y, 1)


# find the best lambda value
def find_best_lamb(lamb_vector):
    l = len(lamb_vector)
    train_err = np.zeros(l)
    val_err = np.zeros(l)
    for i in range(l):
        theta = train_model(poly_X, y, lamb_vector[i])
        train_err[i] = f(theta, poly_X, y, 0)
        val_err[i] = f(theta, poly_X_val, y_val, 0)

    plt.plot(lamb_vector, train_err)
    plt.plot(lamb_vector, val_err)
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.legend(["train", "valiation"])
    plt.show()
    print(train_err)
    print(val_err)


poly_X = ploy_norm_feature(X)
poly_X_val = ploy_norm_feature(X_val)
poly_X_test = ploy_norm_feature(X_test)
# lamb_vec = np.asarray([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
# find_best_lamb(lamb_vec)

theta = train_model(poly_X,y,1)
test_err = f(theta, poly_X_test, y_test, 0)
print(test_err)
