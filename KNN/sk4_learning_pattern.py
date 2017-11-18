from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print("=====data=====")
print(iris_X)
print("===============")
print("data length : " + str(len(iris_X)))
print("====target====")
print(iris_y)
print("===============")
print("target length : " + str(len(iris_y)))
print("===============")
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)