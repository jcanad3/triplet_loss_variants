from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob, torch

class Caltech101(Dataset):
	def __init__(self, mode):
		
		self.transforms = transforms.Compose([
			transforms.Resize((128,128)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]),
			])
		self.mode = mode
		if self.mode == 'train':
			data_path = 'data/train_caltech101/'
		else:
			data_path = 'data/val_caltech101/'

		self.labels = []
		self.img_paths = []
		label = 0
		for img_cat in glob.glob(data_path + '*'):
			for img_path in glob.glob(img_cat + '/*'):
				img = np.array(Image.open(img_path))
				if len(list(img.shape)) == 3 and img.shape[2] == 3:
					self.img_paths.append(img_path)
					self.labels.append(label)
			label += 1
		self.labels = np.array(self.labels)

	def __len__(self):
		return len(self.img_paths)
		

	def __getitem__(self, idx):
		label = self.labels[idx]
		cat_idx = np.argwhere(self.labels == label)

		np.random.shuffle(cat_idx)
		if self.mode == 'train':
			cat_idx = cat_idx[:5]
		else:
			cat_idx = cat_idx[:3]

		cat_idx = cat_idx.flatten()
		imgs = None
		first = True
		for i in cat_idx.tolist():
			img = Image.open(self.img_paths[i])
			img = self.transforms(img)
			
			if first == True:
				imgs = torch.unsqueeze(img, 0)
				first = False
			else:
				img = torch.unsqueeze(img, 0)
				imgs = torch.cat((imgs, img), 0)

		labels = torch.from_numpy(self.labels[cat_idx])

		return imgs, labels
