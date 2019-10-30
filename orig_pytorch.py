from dataset import Caltech101
from torch.utils.data import DataLoader
from models import InceptionTriplet
from batch_all_triplet_loss import batch_all_triplet_loss
from torch.utils.tensorboard import SummaryWriter
#from orig_batch_all_triplet import batch_all_triplet_loss
import torch

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
	batch_size=5,
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
		
		iteration += 1

	# iterate over test
#	model.eval()
#	test_loss = 0
#	test_frac_pos = 0
#	with torch.no_grad():
#		for imgs, labels in val_loader:
#			imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
#			labels = labels.reshape(labels.shape[0]*labels.shape[1],)
#
#			embeddings = model(imgs)
#			b_test_loss, b_test_frac_pos = batch_all_triplet_loss(embeddings, labels, 2.0, 1.0)
#			test_loss += b_test_loss.item()
#			test_frac_pos += b_test_frac_pos.item()
#
#	test_loss /= len(val_loader.dataset)
#	test_frac_pos /= len(val_loader.dataset)
#
#	writer.add_scalar('Loss/test', test_loss)
#	writer.add_scalar('Loss/test_frac_pos', test_frac_pos)
#
#	print('\nTest set: Average loss: {:.4f}, Average Frac Positive: {}\n'.format(test_loss, test_frac_pos))
