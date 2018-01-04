import pandas as pd
import numpy as np
import mxnet as mx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
import logging
import time

logging.basicConfig(level=logging.DEBUG)
py.init_notebook_mode(connected=True)


train = pd.read_json('./data/train/train.json')
print(train.head())
print(train.info())
print(train.describe())
print(train.is_iceberg.value_counts())

def build_x(train_no_test=True):
    if train_no_test:
        x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_1']])
        x_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_2']])
    else:
        x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test['band_1']])
        x_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test['band_2']])
    
    res = np.concatenate([x_band_1[:, :, :, np.newaxis], x_band_2[:, :, :, np.newaxis], ((x_band_1+x_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
    
    return res.transpose(0, 3, 1, 2)

x_train = build_x()
print(x_train.shape)

def plotmy3d(c, name):
    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

# plotmy3d(x_train[2,0,:,:], 'iceberg')
# plotmy3d(x_train[0,0,:,:], 'ship')
# plotmy3d(x_train[2,1,:,:], 'iceberg')
# plotmy3d(x_train[0,1,:,:], 'ship')

# y_train = pd.get_dummies(train.is_iceberg)
y_train = train.is_iceberg
print(y_train.head())
y_train = y_train.values
print(y_train[:5])

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

# should be optional as iterator can take np array as input
x_train = mx.nd.array(x_train)
x_valid = mx.nd.array(x_valid)
y_train = mx.nd.array(y_train)
y_valid = mx.nd.array(y_valid)

print(x_train.shape)
print(type(x_train))
print(y_train.shape)

batch_size = 100
num_epoch = 10
train_iter = mx.io.NDArrayIter(data=x_train, label=y_train, batch_size=batch_size, shuffle=True)
valid_iter = mx.io.NDArrayIter(data=x_valid, label=y_valid, batch_size=batch_size)

print(train_iter.provide_data)
print(train_iter.provide_label)

shape = train_iter.provide_data[0].shape

# network definition

# input
net = mx.sym.var('data')
print('Shapes:')
print(net.infer_shape(data=shape)[1][0])

# normalization
net = mx.sym.BatchNorm(data=net)

# convolution layer 1
print('conv1:')
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=64, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Pooling(data=net, pool_type='max', kernel=(3,3), stride=(2,2))
print(net.infer_shape(data=shape)[1][0])

# convolution layer 2
print('conv2:')
net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=128, pad=(1, 1))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Pooling(data=net, pool_type='max', kernel=(2,2), stride=(2,2))
print(net.infer_shape(data=shape)[1][0])

# convolution layer 3
print('conv3:')
net = mx.sym.Convolution(data=net, kernel=(3,3), num_filter=128, pad=(1,1))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Pooling(data=net, pool_type='max', kernel=(2,2), stride=(2,2))
print(net.infer_shape(data=shape)[1][0])

# convolution layer 4
print('conv4:')
net = mx.sym.Convolution(data=net, kernel=(3,3), num_filter=64, pad=(1,1))
net = mx.sym.Activation(data=net, act_type='relu')
net = mx.sym.Pooling(data=net, pool_type='max', kernel=(2,2), stride=(2,2))
print(net.infer_shape(data=shape)[1][0])

# flatten data
print('flatten:')
net = mx.sym.flatten(net)
print(net.infer_shape(data=shape)[1][0])

# fully connected layers
# hidden layer 1
print('hidden layer 1:')
net = mx.sym.FullyConnected(data=net, num_hidden=512)
net = mx.sym.Activation(data=net, act_type='relu')
print(net.infer_shape(data=shape)[1][0])

# hidden layer 2
print('hidden layer 2:')
net = mx.sym.FullyConnected(data=net, num_hidden=256)
net = mx.sym.Activation(data=net, act_type='relu')
print(net.infer_shape(data=shape)[1][0])

# output layer
print('output layer:')
net = mx.sym.FullyConnected(data=net, num_hidden=2)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.infer_shape(data=shape)[1][0])

# viz
mx.viz.plot_network(symbol=net)

print(net.list_arguments())
print(net.list_outputs())

train_iter.reset()

model = mx.mod.Module(symbol=net, context=mx.gpu(), data_names=['data'], label_names=['softmax_label'])

# model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, inputs_need_grad=False)
# model.init_params(initializer=mx.initializer.Xavier())
# model.init_optimizer(optimizer=mx.optimizer.AdaGrad(rescale_grad=1.0/batch_size))

# ce = mx.metric.CrossEntropy()
# acc = mx.metric.Accuracy()

# for i in range(num_epoch):
#     t = time.time()
#     acc.reset()
#     train_iter.reset()
#     # batch
#     for batch in train_iter:
#         model.forward(batch)
#         print(model.get_outputs())
#         print(batch.label)
#         print(model.get_outputs()[0].shape)
#         print(batch.label[0].shape)
#         # model.update_metric(ce, batch.label)
#         # model.update_metric(acc, batch.label)
#         ce.update(batch.label, model.get_outputs())
#         acc.update(batch.label, model.get_outputs())
#         model.backward()
#         model.update()
#     t = time.time() - t
#     print('Epoch {}/{}:'.format((i+1), num_epoch))
#     print('Cross entropy loss:', ce.get()[1])
#     print('Train accuracy:', acc.get()[1])
#     print()


model.fit(train_data=train_iter,
          eval_data=valid_iter,
          eval_metric=['acc', 'ce'],
          optimizer='adagrad',
          optimizer_params={'learning_rate':0.05},
          initializer=mx.initializer.Xavier(),
          #batch_end_callback=mx.callback.Speedometer(batch_size, 1),
          num_epoch=50)

acc = mx.metric.Accuracy()
ce = mx.metric.CrossEntropy()
model.score(valid_iter, acc)
model.score(valid_iter, ce)
print(acc)
print(ce)
