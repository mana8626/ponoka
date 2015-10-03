import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import matplotlib.pyplot as plt
import pandas.io.data as web
import datetime


# def get_future_value(start_bfe_1, start_bfe_2, start_bfe_3, ..., start_bfe_30):
#     # hogehoge
#     return start_aft_1, start_aft_2, start_aft_3, ..., start_aft_5

# create training and evaluation datas
# n_train_batchset
# n_test_batchset
# x_train
# x_test

start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2014, 9, 1)
nikkei_data= web.DataReader('^N225', 'yahoo', start, end)

values = nikkei_data.values
target_data = []
for i in range(0, len(values)):
    target_data.append(values[i][0])


x_train = []
for j in range(0, (len(values)-35)):
    x_train.append(target_data[j:j+30])
#print(x_train)

t_train = []
for k in range(30, len(values)-5):
    t_train.append(target_data[k:k+5])
#print(t_train)

x_test = x_train
t_test = t_train

x_train = np.array(x_train, dtype=np.float32)
t_train = np.array(t_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
t_test = np.array(t_test, dtype=np.float32)


n_train_batchset = len(x_train)

n_test_batchset = len(x_test)



# define constance
n_input = 30
n_output = 5
n_units = 80

n_epoch = 50

batchsize = 2


# define model
model = FunctionSet(
    l1 = F.Linear(n_input, n_units),
    l2 = F.Linear(n_units, n_output)
)

def forward(x_data, t_data, train = True):
    x, t = Variable(x_data), Variable(t_data)
    h1 = F.dropout(F.relu(model.l1(x)), train = train)
    h2 = F.dropout(model.l2(h1), train = train)
    return F.mean_squared_error(h2, t)


# define training system
optimizer = optimizers.Adam()
optimizer.setup(model)

y_array = []

# training loop
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(n_train_batchset)
    sum_loss = 0
    for i in range(0, n_train_batchset, batchsize):
        x_batch = np.asarray(x_train[perm[i:i + batchsize]])
        t_batch = np.asarray(t_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch, t_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(sum_loss / n_train_batchset))

    y_array.append(sum_loss / n_train_batchset)

    # evaluation
    sum_loss = 0
    for i in range(0, n_test_batchset, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])
        t_batch = np.asarray(t_test[i:i + batchsize])

        loss = forward(x_batch, t_batch, train=False)

        sum_loss += float(loss.data) * len(x_batch)

    print('test  mean loss={}'.format(sum_loss / n_test_batchset))

# do that you want to after training



#x_array = range(1, 51)
#plt.plot(x_array, y_array)
#plt.show()
