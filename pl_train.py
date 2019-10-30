from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_config import InceptionTriplet

model = InceptionTriplet()

trainer = Trainer(val_check_interval=1.0, max_nb_epochs=100)
trainer.fit(model)
