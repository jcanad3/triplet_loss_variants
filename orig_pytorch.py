from dataset import Caltech101
from torch.utils.data import DataLoader
from models import InceptionTriplet
from batch_all_triplet_loss import batch_all_triplet_loss
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def plot_latent(epoch, iteration, embeddings, labels):
	print('Plotting embeddings...')
	umap_embs = UMAP(n_neighbors=10, metric='euclidean').fit_transform(embeddings)
	data = np.column_stack((umap_embs, labels))
	df = pd.DataFrame(data, columns=['UMAP-1', 'UMAP-2', 'label'])
	df['label'] = df['label'].astype(int)

	for lab in np.unique(labels).tolist():
		class_idx = np.argwhere(labels == lab)
		class_embs = umap_embs[class_idx, :]
		class_embs = np.squeeze(class_embs, 1)
		plt.scatter(class_embs[:, 0], class_embs[:, 1], c=np.random.rand(1,3), alpha=0.4)

	plt.title('UMAP of Speaker Embeddings')
	plt.xlabel('UMAP 1')
	plt.ylabel('UMAP 2')
	plt.savefig('plots/' + str(iteration) + '.png')
	plt.close()


writer = SummaryWriter()

train_data = Caltech101(mode='train')
train_loader = DataLoader(
	train_data,
	batch_size=5,
	shuffle=True,
	drop_last=True
)

val_data = Caltech101(mode='val')
val_loader = DataLoader(
	val_data,
	batch_size=32,
	shuffle=True,
	drop_last=True
)

model = InceptionTriplet()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

iteration = 0
for epoch in range(0, 100):
	model.train()
	for batch_idx, (imgs, labels) in enumerate(train_loader):

		imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
		print('Imgs shape', imgs.shape)
		labels = labels.reshape(labels.shape[0]*labels.shape[1],)
		print(labels)	
		optimizer.zero_grad()
		embeddings = model(imgs)
	
		# pass to triplet loss func
		loss, fraction_of_positive = batch_all_triplet_loss(embeddings, labels, 1.0, 1.0)
		
		writer.add_scalar('Loss/train', loss.item(), iteration)
		writer.add_scalar('Loss/train_frac_pos', fraction_of_positive.item(), iteration)

		loss.backward()
		optimizer.step()
		
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Frac_Pos: {}'.format(
			epoch, batch_idx*5, len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item(), fraction_of_positive.item()))
		print('')

		if iteration % 100 == 0:
			embeddings = []
			all_labels = []
			count = 0
			with torch.no_grad():
				for batch_idx, (imgs, labels) in enumerate(val_loader):
					imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
					labels = labels.reshape(labels.shape[0]*labels.shape[1],)

					embs = model(imgs)
					embeddings.append(embs.detach().numpy())
					all_labels.append(labels.detach().numpy().flatten())
					count += len(all_labels)
					if count >= 64:
						break
			embeddings = np.array(embeddings)
			embeddings = embeddings.reshape(embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2])
			all_labels = np.array(all_labels).flatten()
			plot_latent(epoch, iteration, embeddings, all_labels)	
			model.train()
		
		iteration += 1

	torch.save(model.state_dict(), 'ckpts/ckpt_' + str(epoch))

	
