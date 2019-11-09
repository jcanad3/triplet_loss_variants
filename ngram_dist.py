from skimage.util import view_as_windows
import numpy as np
import torch

X = np.random.normal(0, 1, size=(10,9))
X_ngram = []
for idx in range(0, X.shape[0]):
	vec_ngram = []
	for i in range(0, X.shape[1] - 1):
		ngram = X[idx,i:i+3]
		ngram = np.expand_dims(ngram, 0)
		if ngram.shape[1] == 3:
			vec_ngram.append(ngram)
		else:
			ngram = np.append(ngram, np.array([0]))
			ngram = np.expand_dims(ngram, 0)
			vec_ngram.append(ngram)
	X_ngram.append(vec_ngram)

X_ngram = np.array(X_ngram)
X_ngram = np.squeeze(X_ngram, 2)
print(X_ngram.shape)
print(X_ngram)


X_ngram_1 = X_ngram[0, :]
print(X_ngram_1.shape)
X_ngram_2 = X_ngram[1, :]

dists = np.zeros((X_ngram.shape[0], X_ngram.shape[0]))
for i in range(0, X_ngram.shape[0]):
	for j in range(0, X_ngram.shape[0]):
		dists[i,j] = np.sum(np.sum(np.square(X_ngram[i, :, :] - X_ngram[j, :, :]), axis=1), axis=0)
print('Dists mean', np.mean(dists))
print(dists)


l2_2_norm = np.sum(np.square(X[0, :], X[1, :]), axis=0)
print('l2 2 norm', l2_2_norm)


#X_windowed = view_as_windows(X, (1,3), step=1)
#print(X_windowed)

X_tensor = torch.from_numpy(X)
X_rolling = X_tensor.unfold(1, 4, 1)
print(X_rolling)
