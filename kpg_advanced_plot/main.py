from __future__ import print_function

import warnings

warnings.filterwarnings('ignore')

import os

import numpy as np

np.random.seed(42)

# Keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Dataset interfaces
from kgp.datasets.data_utils import data_to_seq

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train

from kgp.models import Model
from kgp.layers import GP

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
from kgp.metrics import mean_absolute_error as MAE
from kgp.metrics import mean_squared_error as MSE
from kgp.metrics import R2

import csv
import matplotlib.pyplot as plt
import re
import numpy as np

import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def csv_to_seq_format(format_list):
    lag_len = 16  # 与参数个数无关
    X = np.array([]).reshape(0, lag_len, 1)
    y = np.array([]).reshape(0, 1, 1)

    for file in format_list:
        X_battery, y_battery = [], []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not np.isnan(float(row[0])):
                    X_battery.append(float(row[0]))
                    y_battery.append(float(row[1]))
        X_battery = np.array(X_battery).reshape(-1, 1)
        y_battery = np.array(y_battery).reshape(-1, 1)
        X_seq, y_seq = data_to_seq(X_battery, y_battery,
                                   t_lag=lag_len, t_future_shift=1, t_future_steps=1, t_sw_step=1)

        X = np.concatenate((X, X_seq), axis=0)
        y = np.concatenate((y, y_seq), axis=0)
    return X, y


def assemble_mlp(input_shape, output_shape, batch_size, nb_train_samples):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(1, activation='relu', name='dense4')(inputs)
    gp = GP(hyp={
        'lik': np.log(0.3),
        'mean': [],
        'cov': [[0.5], [1.0]],
    },
        inf='infGrid', dlik='dlikGrid',
        opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
        mean='meanZero', cov='covSEiso',
        update_grid=1,
        grid_kwargs={'eq': 1, 'k': 70.},
        batch_size=batch_size,
        nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]
    return Model(inputs=inputs, outputs=outputs)


def gp_data(format_list):
    X, y = [], []

    for file in format_list:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not np.isnan(float(row[0])):
                    X.append(float(row[0]))
                    y.append(float(row[1]))
    X = np.array(X).reshape(len(X), 1)
    y = np.array(y).reshape(len(y), 1)
    return X, y


def advanced_plot(data_true, data_predict, data_predict_GPR, data_predict_LSTM, s2, std_predictions_gpr):
    fig_0 = plt.figure(figsize=(4.5, 2.5))
    ax_0 = fig_0.add_axes([0.15, 0.25, 0.8, 0.7])
    ax_0.grid(True)
    ax_0.plot(range(len(data_true)), data_true, color='k', lw=3, label='Ture')
    ax_0.plot(range(len(data_true)), data_predict, color='r', lw=2.5, ls='--',
              label='GPR-LSTM', alpha=0.9)
    ax_0.plot(range(len(data_predict_GPR)), data_predict_GPR, color='c', lw=2, ls='-.',
              label='GPR', alpha=0.9)
    ax_0.plot(range(len(data_predict_LSTM)), data_predict_LSTM, color='m', lw=2, ls=':',
              label='LSTM', alpha=0.9)
    ax_0.fill_between(np.array(range(len(data_predict))), data_predict.squeeze() - 2 * np.sqrt(s2[0].squeeze()),
                      data_predict.squeeze() + 2 * np.sqrt(s2[0].squeeze()), fc='tab:orange', alpha=0.7,
                      label='GP-LSTM 95%CI')
    ax_0.fill_between(np.array(range(len(data_predict_GPR))), data_predict_GPR.squeeze() -  2 * std_predictions_gpr,
                      data_predict_GPR.squeeze() + 2 * std_predictions_gpr, fc='tab:gray', alpha=0.5, label='GP 95%CI')
    leg = plt.legend(loc="lower left", fontsize=8, ncol=2, markerscale=3)
    leg.get_lines()[0].set_linewidth(2)
    leg.get_lines()[1].set_linewidth(2)
    leg.get_lines()[2].set_linewidth(1)
    leg.get_lines()[3].set_linewidth(2)


    ax_0.set_ylim([0.75, 1.02])
    ax_0.set_ylabel('SOH [%]')
    ax_0.set_xlabel('Cycle number')

    error_predict = (data_predict - data_true) / np.max(data_true) * 100
    error_GPR = (data_predict_GPR.ravel() - data_true.ravel()) / np.max(data_true) * 100     # 都转化成数组，避免不期望的广播机制
    error_LSTM = (data_predict_LSTM - data_true) / np.max(data_true) * 100
    x_val = [-2, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    #x_val = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    bar_width = 0.25

    fig_4 = plt.figure(figsize=(4.5, 2.5))
    ax_4 = fig_4.add_axes([0.17, 0.15, 0.78, 0.8])
    bplot = ax_4.boxplot([error_predict, error_GPR, error_LSTM], labels=['GP-LSTM', 'GP', 'LSTM'], patch_artist=True, sym='')
    for patch, color in zip(bplot['boxes'], ['r', 'c', 'm']):
        patch.set_facecolor(color)
    ax_4.set_ylabel('Error [%]')
    ax_4.grid()

    '''
    统计条形图：已完成
    '''
    '''
    inds_predict = np.digitize(error_predict, x_val)
    counts_predict = [np.sum(inds_predict == i) for i in range(len(x_val))]
    fig_1 = plt.figure(figsize=(1.5, 1.6))
    ax_1 = fig_1.add_axes([0.23, 0.25, 0.72, 0.70])
    ax_1.grid(True)
    ax_1.bar(x_val, counts_predict, width=bar_width, color='tab:orange', edgecolor='k')
    ax_1.set_ylabel('Count')
    ax_1.set_xlabel('Error [%]')

    inds_predict = np.digitize(error_GPR, x_val)
    counts_predict = [np.sum(inds_predict == i) for i in range(len(x_val))]
    fig_2 = plt.figure(figsize=(1.5, 1.6))
    ax_2 = fig_2.add_axes([0.23, 0.25, 0.72, 0.70])
    ax_2.grid(True)
    ax_2.bar(x_val, counts_predict, width=bar_width, color='tab:orange', edgecolor='k')
    ax_2.set_ylabel('Count')
    ax_2.set_xlabel('Error [%]')

    inds_predict = np.digitize(error_LSTM, x_val)
    counts_predict = [np.sum(inds_predict == i) for i in range(len(x_val))]

    fig_3 = plt.figure(figsize=(1.5, 1.6))
    ax_3 = fig_3.add_axes([0.23, 0.25, 0.72, 0.70])
    ax_3.grid(True)
    ax_3.bar(x_val, counts_predict, width=bar_width, color='tab:orange', edgecolor='k')
    ax_3.set_ylabel('Count')
    ax_3.set_xlabel('Error [%]')
    '''


    ax_0.tick_params(labelsize=8)
    ax_0.spines['top'].set_linewidth(1.5)
    ax_0.spines['bottom'].set_linewidth(1.5)
    ax_0.spines['left'].set_linewidth(1.5)
    ax_0.spines['right'].set_linewidth(1.5)
    '''
    ax_1.tick_params(labelsize=8)
    ax_1.spines['top'].set_linewidth(1.5)
    ax_1.spines['bottom'].set_linewidth(1.5)
    ax_1.spines['left'].set_linewidth(1.5)
    ax_1.spines['right'].set_linewidth(1.5)
    ax_2.tick_params(labelsize=8)
    ax_2.spines['top'].set_linewidth(1.5)
    ax_2.spines['bottom'].set_linewidth(1.5)
    ax_2.spines['left'].set_linewidth(1.5)
    ax_2.spines['right'].set_linewidth(1.5)
    ax_3.tick_params(labelsize=8)
    ax_3.spines['top'].set_linewidth(1.5)
    ax_3.spines['bottom'].set_linewidth(1.5)
    ax_3.spines['left'].set_linewidth(1.5)
    ax_3.spines['right'].set_linewidth(1.5)
    '''
    ax_4.tick_params(labelsize=8)
    ax_4.spines['top'].set_linewidth(1.5)
    ax_4.spines['bottom'].set_linewidth(1.5)
    ax_4.spines['left'].set_linewidth(1.5)
    ax_4.spines['right'].set_linewidth(1.5)


def main():
    # Load data
    train_dir = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\new_train_2'  # 所选取的训练集
    train_gp_dir = train_dir + '_gp'  # 用于gp的精简数据集
    test_dir = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\test_mannul'  # 选出的效果较好的几个
    test_gp_dir = test_dir + '_gp'
    battery_num = len(os.listdir(train_dir))
    battery_gp_num = len(os.listdir(train_gp_dir))
    batteries_list = os.listdir(train_dir)
    batteries_gp_list = os.listdir(train_gp_dir)
    test_list = os.listdir(test_dir)
    test_gp_list = os.listdir(test_gp_dir)
    batteries_list.sort(key=lambda l: int(re.findall('\d+', l)[1]) + int(re.findall('\d+', l)[0]) * 100)  # 多数段排序
    batteries_gp_list.sort(key=lambda l: int(re.findall('\d+', l)[1]) + int(re.findall('\d+', l)[0]) * 100)

    test_list.sort(key=lambda l: int(re.findall('\d+', l)[1]) + int(re.findall('\d+', l)[0]) * 100)  # 多数段排序
    test_gp_list.sort(key=lambda l: int(re.findall('\d+', l)[1]) + int(re.findall('\d+', l)[0]) * 100)

    '''  个别测试用
    train_format_list = batteries_list[0:-1]
    train_format_gp_list = batteries_gp_list[0:-1]
    test_format_list = batteries_list[battery_num-1:battery_num]
    test_format_gp_list = batteries_gp_list[battery_gp_num-1:battery_gp_num]
    '''

    train_format_list = batteries_list
    train_format_gp_list = batteries_gp_list

    '''turn file into path'''
    train_format_list = [os.path.join(train_dir, i) for i in train_format_list]
    train_format_gp_list = [os.path.join(train_gp_dir, i) for i in train_format_gp_list]

    log_battery_name = []
    log_GP_LSTM_RMSE = []
    log_GP_LSTM_time = []
    log_GP_LSTM_95CI = []

    log_GP_RMSE = []
    log_GP_time = []
    log_GP_95CI = []

    log_LSTM_RMSE = []
    log_LSTM_time = []
    GP_LSTM_first_flag = True
    GP_first_flag = True
    LSTM_first_flag = True
    for (test_format_list, test_format_gp_list) in zip(test_list, test_gp_list):
        log_battery_name.append(test_format_list)

        test_format_list = [test_format_list]
        test_format_gp_list = [test_format_gp_list]
        '''turn file into path'''
        test_format_list = [os.path.join(test_dir, i) for i in test_format_list]
        test_format_gp_list = [os.path.join(test_dir, i) for i in test_format_gp_list]

        X, y = csv_to_seq_format(train_format_list)
        # samples就是样本数量，一般来讲，这个数量和训练结果的数量是一致的，不管是多参量还是单参量，或者是多个时间步长，对应的结果通常是确定的。
        X_test, y_test = csv_to_seq_format(test_format_list)        # to seq操作造成周期减一

        '''开始结束部分会存在较大偏差'''
        X = X[20:-20, :, :]
        y = y[20:-20, :, :]
        X_test = X_test[20:-20, :, :]
        y_test = y_test[20:-20, :, :]

        # Split
        X_train, y_train = X[:], y[:]
        X_test, y_test = X_test[:], y_test[:]
        X_valid, y_valid = X_test[:], y_test[:]

        data = {
            'train': [X_train, y_train],
            'valid': [X_valid, y_valid],
            'test': [X_test, y_test],
        }

        # data = standardize_data(data)
        # Re-format targets
        for set_name in data:
            y = data[set_name][1]
            y = y.reshape((-1, 1, np.prod(y.shape[1:])))
            data[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

        # Model & training parameters
        nb_train_samples = data['train'][0].shape[0]
        input_shape = list(data['train'][0].shape[1:])
        nb_outputs = len(data['train'][1])
        gp_input_shape = (1,)
        batch_size = 128
        epochs = 1000  # 似乎每次异步切换会使误差增大，接近收敛点附近的收敛性较差

        if GP_LSTM_first_flag == True:  # test data 对模型训练没有影响
            '''经过验证，每次循环输入X的长度是1，在最后带有一个（H+X）到X的权重矩阵，其中X=1
            即param的个数：4*[(H+1+1)*H]+H+1 , i.e. 4*H^2+9H+1'''

            nn_params = {
                'H_dim': 128,  # 经测试，Hdim不影响LSTM输出个数, LSTM输出个数为总样本周期数
                'H_activation': 'tanh',
                'dropout': 0.1,
            }
            '''
            gp_params = {
                'cov': 'SEiso',
                'hyp_lik': -2.0,
                #'hyp_cov': [[-0.7], [0.0]],
                'hyp_cov': [[-0.4], [-0.1]],
                'opt': {},
            }
            '''
            gp_params = {
                'cov': 'SEiso',
                # 'hyp_lik': -2.0,
                'hyp_lik': -2.0,
                # 'hyp_cov': [[-0.7], [0.0]],
                'hyp_cov': [[0.0], [0.0]],
                'opt': {'cg_maxit': 20000, 'cg_tol': 1e-4},
                'grid_kwargs': {'eq': 1, 'k': 1e2},
                'update_grid': True,
            }

            # Retrieve model config
            nn_configs = load_NN_configs(filename='lstm.yaml',
                                         input_shape=input_shape,
                                         output_shape=gp_input_shape,
                                         params=nn_params)
            gp_configs = load_GP_configs(filename='gp.yaml',
                                         nb_outputs=nb_outputs,
                                         batch_size=batch_size,
                                         nb_train_samples=nb_train_samples,
                                         params=gp_params)

            # Construct & compile the model
            model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['MSGP']])
            loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
            model.compile(optimizer=Adam(1e-5), loss=loss)

            # Callbacks
            callbacks = [EarlyStopping(monitor='mse', patience=10)]

            # Train the model
            start = time.time()
            history = train(model, data, callbacks=None, gp_n_iter=5,  # gp_n_iter 过大会影响估计精度
                            checkpoint=None, checkpoint_monitor='mse',
                            epochs=epochs, batch_size=batch_size, verbose=1)

            end = time.time()
            GP_LSTM_time = end - start

            # Finetune the model

            model.finetune(*data['train'],
                           batch_size=batch_size,
                           gp_n_iter=100,  # gp_n_iter 过大会影响CI,100是良好的参数
                           verbose=1)

            GP_LSTM_first_flag = False

        # Test the model
        X_test, y_test = data['test']
        y_pre, s2 = list(model.predict(X_test, return_var=True))  # 返回平均值和方差  models.py是kgp的包里面的

        y_pre = np.array(y_pre)
        y_test = np.array(y_test)

        y_pre_plot = y_pre[0, :, 0]
        y_test_plot = y_test[0, :, 0]



        '''GP 不适合超过约10个训练电池'''

        X_gp, y_gp = gp_data(train_format_gp_list)  # 用于MSGP以及GP部分
        X_gp_test, y_gp_test = gp_data(test_format_gp_list)
        X_gp = X_gp[21:-20]         # 补偿to seq的短一损失
        y_gp = y_gp[21:-20]
        X_gp_test = X_gp_test[21:-20]
        y_gp_test = y_gp_test[21:-20]


        if GP_first_flag == True:
            # Error standard deviation.
            sigma_n = 0.4
            # Define kernel parameters.
            l = 0.1
            sigma_f = 2

            # Define kernel object.
            kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-3, 1e3)) \
                     * RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=10)

            start = time.time()

            gp.fit(X_gp, y_gp)

            end = time.time()
            gp_time = end - start

            GP_first_flag = False

        y_gp_pred, std_predictions_gpr = gp.predict(X_gp_test, return_std=True)     # 会多出一维

        rmse_predict = RMSE(y_gp_test, y_gp_pred)
        print('GP Test RMSE:', rmse_predict)


        log_GP_RMSE.append(rmse_predict)
        log_GP_time.append(gp_time)
        GP_95CI = np.mean(std_predictions_gpr)
        log_GP_95CI.append(GP_95CI)
        print('GP_95CI: ', GP_95CI)
        '''LSTM'''
        data = {
            'train': [X_train, y_train],
            'valid': [X_valid, y_valid],
            'test': [X_test, y_test],
        }
        # no need for reformat

        input_shape = list(data['train'][0].shape[1:])
        output_shape = list(data['train'][1].shape[1:])
        batch_size = 128
        epochs = 200

        if LSTM_first_flag == True:
            nn_params = {
                'H_dim': 128,
                'H_activation': 'tanh',
                'dropout': 0.1,
            }

            # Retrieve model config
            configs = load_NN_configs(filename='lstm.yaml',
                                      input_shape=input_shape,
                                      output_shape=output_shape,
                                      params=nn_params)

            # Construct & compile the model
            model3 = assemble('LSTM', configs['1H'])
            model3.compile(optimizer=Adam(1e-5), loss='mse')
            # Callbacks
            callbacks = [EarlyStopping(monitor='val_loss', patience=20)]

            # Train the model
            start = time.time()
            history = train(model3, data, callbacks=None,
                            epochs=epochs, batch_size=batch_size, verbose=1)

            end = time.time()
            LSTM_time = end - start

            LSTM_first_flag = False

        # Test the model
        X_test, y_test = data['test']
        y_preds = model3.predict(X_test).squeeze()

        rmse_predict = RMSE(y_test, y_preds)
        print('LSTM Test RMSE:', rmse_predict)

        print('GP_LSTM_time:', GP_LSTM_time, '\nGP_time: ', gp_time, '\nLSTM_time: ', LSTM_time)

        log_LSTM_RMSE.append(rmse_predict)
        log_LSTM_time.append(LSTM_time)

        print('battery name: ', test_format_list)

        advanced_plot(y_test_plot, y_pre_plot, y_gp_pred, y_preds, s2, std_predictions_gpr)
    plt.show()


if __name__ == '__main__':
    main()
