from Trainer import Trainer

## directory of data
path = "./data/FinalBalancedDataset.csv"

## Models
# put any of the models in the brackets into the list {svm, simple, lstm}
# you can choose one or more of the models to train
models = ['svm', 'simple', 'lstm']

## HyperParms
bs = 64
embed = 150
hdim = 150
lr = 0.00005
epochs = 30
gam= 0.96
tr = 0.9
ts= 0.1
early = 4
layers = 4
thresh = 5
most = 29
dropout = 0.44
bi = True

model = Trainer(path, models, bs, embed, hdim, lr, epochs, gam, tr, ts, early, layers, thresh, most, dropout, bi)
model.run()