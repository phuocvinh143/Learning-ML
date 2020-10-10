import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris['data']
iris_y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def acc_cal(x, y):
    return np.sum(x == y)

print(acc_cal(y_pred, y_test)/50)