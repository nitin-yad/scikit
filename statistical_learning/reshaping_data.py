from sklearn import datasets

digits = datasets.load_digits()
# dataset is made of 1797 8*8 images
print(digits.images.shape)

import matplotlib.pyplot as plt
# gray_r is inverse grayscale
plt.imshow(digits.images[-1], cmap='gray_r')
plt.show()
data = digits.images.reshape((digits.images.shape[0], -1))