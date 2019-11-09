from models import InceptionTriplet
from dataset import Caltech101
from torch.utils.data import DataLoader
from umap import UMAP
import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch


val_data = Caltech101(mode='val')
val_loader = DataLoader(
	val_data,
	batch_size=100,
	shuffle=True,
)

model = InceptionTriplet()
model.load_state_dict(torch.load('ckpts/ckpt_3'))
model.eval()

img_labels = []
embs = []

with torch.no_grad():
	for imgs, labels in val_loader:
		imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
		labels = labels.reshape(labels.shape[0]*labels.shape[1],)
		img_labels.append(labels.detach().numpy())
		embeddings = model(imgs)
		embs.append(embeddings.detach().numpy())
		print('Size labels', len(img_labels))
		print('Size embs', len(embs))

		if len(labels) >= 200:
			break

embs = np.array(embs)
embs = np.squeeze(embs, 0)
print('embs shape', embs.shape)
img_labels = np.array(img_labels)
img_labels = img_labels.T
img_labels = img_labels.astype(int)
print('img_labels shape', img_labels.shape)

umap_embs = UMAP(n_neighbors=10, metric='manhattan').fit_transform(embs)

data = np.column_stack((umap_embs, img_labels))

df = pd.DataFrame(data, columns=['umap-1', 'umap-2', 'labels'])
df['labels'] = pd.to_numeric(df['labels'], downcast='integer')
print(df.head(10))

fig = px.scatter(df, x='umap-1', y='umap-2', color='labels', hover_data=['labels'])
fig.show()
#ax = sns.scatterplot(x='umap-1', y='umap-2', data=df, hue='labels')p
#plt.show()

