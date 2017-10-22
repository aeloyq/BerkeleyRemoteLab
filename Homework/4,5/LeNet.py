from nnbuilder import *

### Set Global Config ###
config.set(max_epoch=100, valid_batch_size=5000,log=False)

### Set Optimizer Config ###
gradientdescent.sgd.learning_rate = 0.2

### Set Extension Config ###
monitor.config.set(plot=True)
earlystop.config.set(patience=100, valid_freq=500, valid_epoch=False)
saveload.config.set(save_freq=10000,load=False)


### Build Model ###
def get_model(drop=False):
    LeNet = model((1, 28, 28), X=var.X.image)
    LeNet.add(conv(nfilters=6, filtersize=[5, 5]))
    LeNet.add(subsample(windowsize=(2, 2)))
    LeNet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]))
    LeNet.add(subsample(windowsize=(2, 2)))
    LeNet.add(conv(nfilters=120, filtersize=[4, 4]))
    LeNet.add(flatten())
    LeNet.add(hnn(84))
    if drop:
        LeNet.add(dropout(0.5))
    LeNet.add(softmax(10))
    if drop:
        LeNet.add(dropout(0.5))
    return LeNet


### Load Data ###
data = tools.loaddatas.Load_mnist_image("./datasets/mnist.pkl.gz")

### Fit Model ###
batch_size_list = []

print("\nBegin Training")
### try batch size from 1024 to 4096
print('Trying Diferent Batch Size ...')
for i in [1024, 2048, 4096]:
    name = 'LeNet5-batch-{}'.format(i)
    config.set(name=name, batch_size=i)
    monitor.config.set(compare_repo=batch_size_list)
    train(data=data, model=get_model(), optimizer=gradientdescent.sgd,
          extensions=[monitor, earlystop, saveload])
    batch_size_list.append(name)
    print("   ...")

### try 0.01 lr ###
name = 'LeNet5-lr-0.1'
print('Trying {} ...'.format(name))
config.set(name=name, batch_size=2048)
gradientdescent.sgd.learning_rate = 0.1
train(data=data, model=get_model(), optimizer=gradientdescent.sgd,
      extensions=[monitor, earlystop, saveload])
batch_size_list.append(name)

### try nadam ###
name = 'LeNet5-nadam'
print('Trying {} ...'.format(name))
config.set(name=name)
gradientdescent.nadam.learning_rate = 0.01
train(data=data, model=get_model(), optimizer=gradientdescent.nadam,
      extensions=[monitor, earlystop, saveload])
batch_size_list.append(name)

### try dropout ###
name = 'LeNet5-dropout-nadam'
print('Trying {} ...'.format(name))
config.set(name=name)
train(data=data, model=get_model(True), optimizer=gradientdescent.nadam,
      extensions=[monitor, earlystop, saveload])
print("\nOK !")