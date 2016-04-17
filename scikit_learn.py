import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

#print'1=',(digits.data[0])
#print'2=',(digits.target[0:20])
#print'3=',(digits.images[0])

clf = svm.SVC(gamma=0.0001, C=100)

x, y = digits.data[:-10], digits.target[:-10]
clf.fit(x, y)

print ('Prediction:', clf.predict(digits.data[-6]))
plt.imshow(digits.images[-6], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
