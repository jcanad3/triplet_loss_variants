from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_config import InceptionTriplet

model = InceptionTriplet()

trainer = Trainer(max_nb_epochs=100)
trainer.fit(model)
