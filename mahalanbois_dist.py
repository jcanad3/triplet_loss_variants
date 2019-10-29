from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import DistanceMetric
from sklearn.covariance import EmpiricalCovariance as EC
from batch_all_triplet_loss import _Lp_pairwise_distances, _mahalanobis_dist, _batch_mahalanobis_dist
import numpy as np
import torch

X = np.abs(np.random.normal(0, 10, (10, 10)))
print('X', X)

y = []
count = 0
for i in range(0,2):
	for j in range(0,5):
		y.append(count)
	count += 1

print(y)

#class_measures = []
#for i in range(0, 10, 5):
#	class_data = X[i:i+5, :]
#	ec = EC().fit(class_data)
#	class_measures.append(ec)
#
#
##dist = ec.mahalanobis(X)
##print(dist)
#
##dist = mahalanobis(X[1,:], X[2,:], VI=ec.precision_)
##print('Dist', dist)
#
## calc pairwise distances
## choose cov based on label in y
#distances = np.zeros((X.shape[0], X.shape[0]))
#for i in range(0, X.shape[0]):
#	i_label = y[i]
#	i_mean = class_measures[i_label].location_
#	i_precis = class_measures[i_label].precision_
#	for j in range(0, X.shape[1]):
#		distances[i,j] = np.sqrt(np.matmul(np.matmul((X[i,j] - i_mean), i_precis), (X[i,j] - i_mean).T))
#
#print(distances)

m_dist = _mahalanobis_dist(torch.from_numpy(X), torch.from_numpy(np.array(y)))
print(m_dist)

print('Batch Maha')
print(_batch_mahalanobis_dist(torch.from_numpy(X)))


print('')
pairwise_dist = _Lp_pairwise_distances(torch.from_numpy(X), 2.0)
print(pairwise_dist)
