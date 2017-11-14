import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100,10)
#print(X)
y = rng.binomial(1, 0.5, 100)
#print(y)
X_test = rng.rand(5,10)

clf = SVC()

clf.set_params(kernel='linear').fit(X,y)
print(clf.predict(X_test))

clf.set_params(kernel='rbf').fit(X,y)
print(clf.predict(X_test))
