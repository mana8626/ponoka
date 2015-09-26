import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import matplotlib.pyplot as plt


# def get_future_value(start_bfe_1, start_bfe_2, start_bfe_3, ..., start_bfe_30):
#     # hogehoge
#     return start_aft_1, start_aft_2, start_aft_3, ..., start_aft_5

# create training and evaluation datas
# n_train_batchset
# n_test_batchset
# x_train
# x_test

x_train = [
    [
        16147.540039,
        15835.410156,
        15943.679688,
        16002.879883,
        15785.150391,
        15912.059570,
        15657.200195,
        15649.070312,
        15845.150391,
        15695.459961,
        15724.139648,
        15710.889648,
        15749.009766,
        15900.629883,
        15473.570312,
        15091.450195,
        15038.639648,
        15164.339844,
        15112.700195,
        15132.230469,
        14788.559570,
        14353.330078,
        14213.099609,
        14233.419922,
        14387.110352,
        14647.830078,
        14821.730469,
        14785.839844,
        14538.200195,
        14343.730469
    ],
    [
        16147.540039,
        16147.540039,
        15835.410156,
        15943.679688,
        16002.879883,
        15785.150391,
        15912.059570,
        15657.200195,
        15649.070312,
        15845.150391,
        15695.459961,
        15724.139648,
        15710.889648,
        15749.009766,
        15900.629883,
        15473.570312,
        15091.450195,
        15038.639648,
        15164.339844,
        15112.700195,
        15132.230469,
        14788.559570,
        14353.330078,
        14213.099609,
        14233.419922,
        14387.110352,
        14647.830078,
        14821.730469,
        14785.839844,
        14538.200195
    ]
]

t_train = [
    [
        14514.469727,
        14729.480469,
        14701.139648,
        14618.610352,
        14803.639648
    ],
    [
        14343.730469,
        14514.469727,
        14729.480469,
        14701.139648,
        14618.610352
    ]
]

x_test = x_train
t_test = t_train

x_train = np.array(x_train, dtype=np.float32)
t_train = np.array(t_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
t_test = np.array(t_test, dtype=np.float32)


n_train_batchset = 2

n_test_batchset = 2



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
x_array = range(1, 51)
plt.plot(x_array, y_array)
plt.show()
