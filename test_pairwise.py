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

a = torch.randn((5,10))

lp_dist = _Lp_pairwise_distances(a, 2)
print('lp_dist', torch.pow(lp_dist, 2))

euc_pairwise = _pairwise_distances(a)
print('Euc pairwise', euc_pairwise)
