from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


def find_by_equation(X, y):
    # Building Xbar
    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis=1)

    # Calculating weights of the fitting line
    A = np.dot(Xbar.T, Xbar)
    b = np.dot(Xbar.T, y)
    w = np.dot(np.linalg.pinv(A), b)
    print('w = ', w)

    # Preparing the fitting line
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(145, 185, 2)
    y0 = w_0 + w_1 * x0

    # Drawing the fitting line
    plt.plot(X.T, y.T, 'ro')  # data
    plt.plot(x0, y0)  # the fitting line
    plt.axis([140, 190, 45, 75])
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()

    height = [162, 169]
    weight_predict = [w_0 + w_1 * x for x in height]
    print(weight_predict)


def find_by_module(X, y):
    # Building Xbar
    one = np.ones((X.shape[0], 1))
    Xbar = np.concatenate((one, X), axis=1)

    # fit the model by Linear Regression
    regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculating the bias
    regr.fit(Xbar, y)

    height = [162, 169]
    weight_predict = [regr.coef_[0][0] + regr.coef_[0][1] * x for x in height]
    print(weight_predict)


# height (cm)
X = np.array([[147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kq)
y = np.array([[55, 56, 57, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73]]).T

find_by_equation(X, y)
find_by_module(X, y)
