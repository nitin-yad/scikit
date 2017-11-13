
from sklearn import datasets

#loading iris and digits dataset
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

print(digits.target)
