from Trainer import Trainer

# directory of data
path = "./data/FinalBalancedDataset.csv"

## HyperParms
bs = 50
embed = 200
hdim = 100
lr = 0.00005
epochs = 20
gam= 0.96
tr = 0.9
ts= 0.1
early = 3
layers = 2
thresh = 12
most = 40
dropout = 0.5
bi = True

model = Trainer(path, bs, embed, hdim, lr, epochs, gam, tr, ts, early, layers, thresh, most, dropout, bi)
model.run()