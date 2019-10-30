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
	indices_equal = data.type(torch.BoolTensor)

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

def batch_all_triplet_loss(embeddings, labels, squared=True):
	margin = 1.0	
	pairwise_dist = _pairwise_distances(embeddings, squared=squared)

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

def gaussian_embeddings(embeddings, recording_labels, species_labels):
	embeddings = embeddings.detach().numpy()
	recording_labels = recording_labels.detach().numpy()
	species_labels = species_labels.detach().numpy()

	uniq_rec_labels = np.unique(recording_labels).tolist()

	gaussian_embeddings = np.asarray(())
	gaussian_labels = np.array([])

	# get all the embeddings for individual recording
	for label in uniq_rec_labels:
		rec_idxs = np.argwhere(label == recording_labels)
		species_id = np.unique(species_labels[rec_idxs])[0]
		species_id = int(species_id)
		
		rec_embs = embeddings[rec_idxs, :]
		rec_embs = rec_embs.reshape(rec_embs.shape[0]*rec_embs.shape[1], rec_embs.shape[2])

		# fit gm on recording embeddings
		gm = GM(n_components=1, covariance_type='full').fit(rec_embs)
	
		# samples gm 10 times
		gm_samples = gm.sample(n_samples=10)[0]
		if gaussian_embeddings.size == 0:
			gaussian_embeddings = gm_samples
		else:
			gaussian_embeddings = np.vstack((gaussian_embeddings, gm_samples))

		# append correct species_labels to each of the 10 embeddings
		label_repeats = np.repeat(species_id, 10)
		gaussian_labels = np.append(gaussian_labels, label_repeats)

	gaussian_labels = gaussian_labels.astype(int)

	# convert them back to tensors
	gaussian_embeddings = torch.tensor(gaussian_embeddings, requires_grad=True)
	gaussian_labels = torch.from_numpy(gaussian_labels)

	# update data types
	gaussian_embeddings = gaussian_embeddings.type(torch.FloatTensor)

	return gaussian_embeddings, gaussian_labels

def mean_embeddings(embeddings, recording_labels, species_labels):
	embeddings = embeddings.detach().numpy()
	recording_labels = recording_labels.detach().numpy()
	species_labels = species_labels.detach().numpy()

	uniq_rec_labels = np.unique(recording_labels).tolist()

	mean_embeddings = np.asarray(())
	mean_labels = np.array([])

	# get all the embeddings for individual recording
	for label in uniq_rec_labels:
		rec_idxs = np.argwhere(label == recording_labels)
		species_id = np.unique(species_labels[rec_idxs])[0]
		species_id = int(species_id)
		
		rec_embs = embeddings[rec_idxs, :]
		rec_embs = rec_embs.reshape(rec_embs.shape[0]*rec_embs.shape[1], rec_embs.shape[2])

		# find mean of embeddings
		mean_emb = np.mean(rec_embs, axis=0)

		# samples gm 10 times
		for noise in range(10):
			# add small amount of gaussian noise
			mean_w_noise = mean_emb + np.random.normal(0, 0.1, 1)
			if mean_embeddings.size == 0:
				mean_embeddings = mean_w_noise
			else:
				mean_embeddings = np.vstack((mean_embeddings, mean_w_noise))

		# append correct species_labels to each of the 10 embeddings
		label_repeats = np.repeat(species_id, 10)
		mean_labels = np.append(mean_labels, label_repeats)

	mean_labels = mean_labels.astype(int)

	# convert them back to tensors
	mean_embeddings = torch.tensor(mean_embeddings, requires_grad=True)
	mean_labels = torch.from_numpy(mean_labels)

	# update data types
	mean_embeddings = mean_embeddings.type(torch.FloatTensor)

	return mean_embeddings, mean_labels

def batch_all_gaussian_triplet_loss(embeddings, recording_labels, species_labels, squared=True):
	# fit gaussian mixture to embeddings with same recording_labels
	#embeddings, labels = gaussian_embeddings(embeddings, recording_labels, species_labels)
	embeddings, labels = mean_embeddings(embeddings, recording_labels, species_labels)

	margin = 1.0	
	pairwise_dist = _pairwise_distances(embeddings, squared=squared)

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
