from fastai.vision import *
from fastai.metrics import error_rate
from torchsummary import summary

data = ImageDataBunch.from_folder("./data", valid_pct=0.2)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
print(learn.summary())
# learn.lr_find()
# learn.recorder.plot()