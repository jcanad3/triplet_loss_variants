from sklearn.mixture import GaussianMixture as GM
import numpy as np
import torch

def _pairwise_distances(embeddings, squared=True):
 	# Get the dot product between all embeddings
	# shape (batch_size, batch_size)
	dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

	# Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
	# This also provides more numerical stability (the diagonal of the result will be exactly 0).
	# shape (batch_size,)
	square_norm = torch.diagonal(dot_product)

	# Compute the pairwise distance matrix as we have:
	# ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
	# shape (batch_size, batch_size)
	distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

	# Because of computation errors, some distances might be negative so we put everything >= 0.0
	distances = torch.max(distances, torch.tensor([0.0]))

	if not squared:
		# Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
		# we need to add a small epsilon where distances == 0.0
		#zeros_mask = torch.zeros([distances
		mask = torch.eq(distances, torch.tensor([0.0]))
		distances = distances + mask * 1e-16

		distances = torch.sqrt(distances)

		# Correct the epsilon added: set the distances on the mask to be exactly 0.0
		distances = distances * (1.0 - mask)

	return distances

def _Lp_pairwise_distances(embeddings, p):
	distances = torch.zeros(embeddings.shape[0], embeddings.shape[0])
	for i in range(0, embeddings.shape[0]):
		for j in range(0, embeddings.shape[0]):
			distances[i,j] = torch.pow(torch.sum(torch.pow(embeddings[i,:] - embeddings[j,:], p)), 1/p)

	return distances

def _get_anchor_positive_triplet_mask(labels):
	"""Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
	Args:
		labels: tf.int32 `Tensor` with shape [batch_size]
	Returns:
		mask: tf.bool `Tensor` with shape [batch_size, batch_size]
	"""
	# Check that i and j are distinct
	indices_equal = torch.eye(labels.shape[0])
	indices_not_equal = ~indices_equal

	# Check if labels[i] == labels[j]
	# Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
	labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

	# Combine the two masks
	mask = indices_not_equal & labels_equal

	return mask


def _get_anchor_negative_triplet_mask(labels):
	"""Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
	Args:
		labels: tf.int32 `Tensor` with shape [batch_size]
	Returns:
		mask: tf.bool `Tensor` with shape [batch_size, batch_size]
	"""
	# Check if labels[i] != labels[k]
	# Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
	labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

	mask = ~labels_equal

	return mask


def _get_triplet_mask(labels):
	"""Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
	A triplet (i, j, k) is valid if:
		- i, j, k are distinct
		- labels[i] == labels[j] and labels[i] != labels[k]
	Args:
		labels: tf.int32 `Tensor` with shape [batch_size]
	"""
	# Check that i, j and k are distinct
	data = torch.eye(labels.shape[0])
	# changing data type for logical operations
	indices_equal = data.type(torch.ByteTensor)

	indices_not_equal = ~indices_equal
	
	i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
	i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
	j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

	distinct_indices = ((i_not_equal_j & i_not_equal_k) & j_not_equal_k)


	# Check if labels[i] == labels[j] and labels[i] != labels[k]
	label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
	i_equal_j = torch.unsqueeze(label_equal, 2)
	i_equal_k = torch.unsqueeze(label_equal, 1)

	valid_labels = i_equal_j & (~i_equal_k)

	# Combine the two masks
	byte_mask = distinct_indices & valid_labels
	mask = byte_mask.type(torch.FloatTensor)

	return mask

def batch_all_triplet_loss(embeddings, labels, p, margin, squared=True):
	#pairwise_dist = _pairwise_distances(embeddings, squared=squared)
	pairwise_dist = _Lp_pairwise_distances(embeddings, p)

	anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
	anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

	# Compute a 3D tensor of size (batch_size, batch_size, batch_size)
	# triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
	# Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
	# and the 2nd (batch_size, 1, batch_size)
	triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

	# Put to zero the invalid triplets
	# (where label(a) != label(p) or label(n) == label(a) or a == p)
	mask = _get_triplet_mask(labels)
	#mask = tf.to_float(mask)
	triplet_loss = torch.mul(mask, triplet_loss)

	# Remove negative losses (i.e. the easy triplets)
	triplet_loss = torch.max(triplet_loss, torch.tensor([0.0]))

	# Count number of positive triplets (where triplet_loss > 0)
	valid_triplets = torch.gt(triplet_loss, 1e-16)
	num_positive_triplets = torch.sum(valid_triplets)
	num_valid_triplets = torch.sum(mask)
	fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

	# Get final mean triplet loss over the positive valid triplets
	triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

	return triplet_loss, fraction_positive_triplets
