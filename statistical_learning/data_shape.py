from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
# (150, 4) where 150 is the sample size, each described by 4 features
print(data.shape)