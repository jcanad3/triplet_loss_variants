from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import Caltech101
from batch_all_triplet_loss import batch_all_triplet_loss
import torch.nn as nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchvision.models as models
import torch, os

class InceptionTriplet(pl.LightningModule):
	def __init__(self):
		super(InceptionTriplet, self).__init__()
		self.resnet = models.resnet18(pretrained=True)
		for param in self.resnet.parameters():
			param.requires_grad = False

		out_features = self.resnet.fc.in_features
		# add embedding layer
		self.resnet.fc = nn.Sequential(
			nn.Linear(out_features, 1000),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(1000, 128)
		)

		# p is the value for Lp norm
		self.p_norm = 0.5

		# margin for triplet loss
		self.margin = 0.5

	def forward(self, x):
		return self.resnet(x)

	def training_step(self, batch, batch_nb):
		imgs, labels = batch
		imgs = torch.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]))
		labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1],))

		embeddings = self.forward(imgs)
		
		loss, fraction_of_positive = batch_all_triplet_loss(embeddings, labels, self.p_norm, self.margin)
		tensorboard_logs = {'train_loss': loss, 'train_frac_positive': fraction_of_positive}
		
		return {'loss': loss, 'log': tensorboard_logs}

	def validation_step(self, batch, batch_nb):
		imgs, labels = batch
		imgs = torch.reshape(imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]))
		labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1],))

		embeddings = self.forward(imgs)

		loss, fraction_of_positive = batch_all_triplet_loss(embeddings, labels, self.p_norm, self.margin)
		
		return {'val_loss': loss, 'val_frac_positive': fraction_of_positive}

	def validation_end(self, outputs):
		avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		avg_val_frac_positive = torch.stack([x['val_frac_positive'] for x in outputs]).mean()
		tensorboard_logs = {'avg_val_loss': avg_val_loss, 'avg_val_frac_positive': avg_val_frac_positive}
		return {'avg_val_loss': avg_val_loss, 'log': tensorboard_logs}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-3)

	@pl.data_loader
	def train_dataloader(self):	
		train_data = Caltech101(mode='train')
		
		data_loader = DataLoader(
			train_data,
			batch_size=5,
			shuffle=True,
			drop_last=True
		)
		
		return data_loader				
		
	@pl.data_loader
	def val_dataloader(self):
		val_data = Caltech101(mode='val')
		
		data_loader = DataLoader(
			val_data,
			batch_size=5,
			shuffle=True,
			drop_last=True
		)

		return data_loader
