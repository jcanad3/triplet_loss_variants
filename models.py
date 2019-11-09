import torchvision.models as models
import torch.nn as nn

class InceptionTriplet(nn.Module):
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
#			nn.Dropout(0.2),
			nn.Linear(1000, 128)
		)

		# p is the value for Lp norm
		self.p_norm = 1.0

		# margin for triplet loss
		self.margin = 1.0

	def forward(self, x):
		return self.resnet(x)
