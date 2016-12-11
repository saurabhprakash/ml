from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# DecisionTreeClassifier
from sklearn import tree
classifier_1 = tree.DecisionTreeClassifier()

classifier_1.fit(X_train, y_train)
predictions = classifier_1.predict(X_test)

from sklearn.metrics import accuracy_score
print 'DecisionTreeClassifier accuracy_score=', accuracy_score(y_test, predictions)


# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
classifier_2 = KNeighborsClassifier()

classifier_2.fit(X_train, y_train)
predictions = classifier_2.predict(X_test)

from sklearn.metrics import accuracy_score
print 'KNeighborsClassifier accuracy_score=', accuracy_score(y_test, predictions)
