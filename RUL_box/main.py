"""MSGP-LSTM regression on Actuator data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from six.moves import xrange

import numpy as np

np.random.seed(42)
# Keras
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# Dataset interfaces
from kgp.datasets.sysid import load_data
from kgp.datasets.data_utils import data_to_seq, standardize_data

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE

import os
import re
import csv

plt.rcParams['font.family']='Times New Roman'
plt.rcParams ['legend.fontsize'] = 9
def standardize_input_data(X_train, X_test, X_valid):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid

def normalize_input_data(X_train, X_test, X_valid):
    X_max = np.max(X_train, axis=0)
    X_min = np.min(X_train, axis=0)

    # X_train = (X_train - X_min) / (X_max - X_min)
    # X_test = (X_test - X_min) / (X_max - X_min)
    # X_valid = (X_valid - X_min) / (X_max - X_min)
    X_train = (X_train - 0.9) / 0.2
    X_test = (X_test - 0.9) / 0.2
    X_valid = (X_valid - 0.9) / 0.2

    return X_train, X_test, X_valid


def main():
    SOH_series_dir = r'C:\Users\iCosMea Pro\My paper\python projects\RUL_data\train_SOH_2'
    SOH_series_list = os.listdir(SOH_series_dir)
    SOH_series_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    SOH_series_dir_test = r'C:\Users\iCosMea Pro\My paper\python projects\RUL_data\test_SOH_two_step_MCC'
    SOH_series_list_test = os.listdir(SOH_series_dir_test)
    SOH_series_list_test.sort(key=lambda l: int(re.findall('\d+', l)[1]))

    t_lag = 8
    lr = 1e-4
    lstm_lr = 1e-4
    t_sw_step = 10
    gp_lstm_epoch = 1000
    LSTM_epochs = 500
    gp_lstm_dim = 256
    LSTM_dim = 256
    frac = 1000
    '''train set'''
    X_raw = np.array([]).reshape(-1, 1)
    y_raw = np.array([]).reshape(-1, 1)
    y_raw_LSTM = np.array([]).reshape(-1, 1)
    for batt in SOH_series_list:
        with open(os.path.join(SOH_series_dir, batt)) as f:
            batt_full_SOH_series = []
            y_batt, y_batt = [], []
            reader = csv.reader(f)
            for row in reader:
                batt_full_SOH_series.append(float(row[0]))
            # total_lifespan = len(batt_full_SOH_series) + 1
            EOL = np.argwhere(np.array(batt_full_SOH_series) < 0.801)
            if EOL.size == 0:
                continue
            batt_use_SOH_series=batt_full_SOH_series[:EOL[0,0]]
            X_batt = np.array(batt_use_SOH_series).reshape(-1, 1)
            y_batt = np.flip(np.arange(EOL[0])).reshape(-1, 1)
            y_batt_LSTM = np.flip(np.arange(EOL[0]) / frac).reshape(-1, 1)
        X_raw = np.concatenate((X_raw, X_batt), axis=0)
        y_raw = np.concatenate((y_raw, y_batt), axis=0)
        y_raw_LSTM = np.concatenate((y_raw_LSTM, y_batt_LSTM), axis=0)
    X_train, y_train = data_to_seq(X_raw, y_raw, t_lag=t_lag, t_future_shift=1, t_future_steps=1, t_sw_step=t_sw_step)
    _, y_train_LSTM = data_to_seq(X_raw, y_raw_LSTM, t_lag=t_lag, t_future_shift=1, t_future_steps=1, t_sw_step=t_sw_step)

    '''test set'''
    X_set = []
    y_set = []
    y_set_LSTM = []
    for batt in SOH_series_list_test:
        with open(os.path.join(SOH_series_dir_test, batt)) as f:
            batt_full_SOH_series = []
            y_batt, y_batt = [], []
            reader = csv.reader(f)
            for row in reader:
                batt_full_SOH_series.append(float(row[0]))
            # total_lifespan = len(batt_full_SOH_series) + 1
            EOL = np.argwhere(np.array(batt_full_SOH_series) < 0.801)
            if EOL.size == 0:
                continue
            batt_use_SOH_series = batt_full_SOH_series[:EOL[0, 0]]
            X_batt = np.array(batt_use_SOH_series).reshape(-1, 1)
            y_batt = np.flip(np.arange(EOL[0])).reshape(-1, 1)
            y_batt_LSTM = np.flip(np.arange(EOL[0]) / frac).reshape(-1, 1)
            X_test_batt, y_test_batt = data_to_seq(X_batt, y_batt, t_lag=t_lag, t_future_shift=1, t_future_steps=1, t_sw_step=t_sw_step)
            _, y_test_batt_LSTM = data_to_seq(X_batt, y_batt_LSTM, t_lag=t_lag, t_future_shift=1, t_future_steps=1,
                                                   t_sw_step=t_sw_step)
        X_set.append(X_test_batt)
        y_set.append(y_test_batt)
        y_set_LSTM.append(y_test_batt_LSTM)
    # Load data
    # Split


    X_train, _, _ = normalize_input_data(X_train, X_train, X_train)
    X_set_std = []
    for i in X_set:
        _i, _, _ = normalize_input_data(i, i, i)
        X_set_std.append(_i)
    # y_train, _, _, y_max = normalize_input_data(y_train, y_train, y_train)
    '''取一个训练集，用于训练时表示'''    # two_stage 22 as example
    test_cell = -4
    data = {
        'train': [X_train, y_train],
        'valid': [X_set_std[test_cell], y_set[test_cell]],
        'test': [X_set_std[test_cell], y_set[test_cell]],
    }

    data_LSTM = {
        'train': [X_train, y_train_LSTM],
        'valid': [X_set_std[test_cell], y_set_LSTM[test_cell]],
        'test': [X_set_std[test_cell], y_set_LSTM[test_cell]],
    }
    #assert False

    # Re-format targets
    for set_name in data:
        y = data[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data[set_name][1] = [y[:, :, i] for i in xrange(y.shape[2])]

    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = data['train'][0].shape[1:]
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 16
    epochs = gp_lstm_epoch

    nn_params = {
        'H_dim': gp_lstm_dim,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }

    gp_params = {
        'cov': 'SEiso',
        'hyp_lik': -2.0,
        'hyp_cov': [[-0.7], [0.0]],
        'opt': {'cg_maxit': 500, 'cg_tol': 1e-4},
        'grid_kwargs': {'eq': 1, 'k': 1e2},
        'update_grid': True,
    }
    '''
    gp_params = {
        'cov': 'SEiso',
        'hyp_lik': -2.0,
        # 'hyp_cov': [[-0.7], [0.0]],
        'hyp_cov': [[-0.4], [-0.1]],
        'opt': {},
    }
    '''
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
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(lr), loss=loss)
    model.summary()

    # Callbacks
    callbacks = [EarlyStopping(monitor='nlml', patience=10)]

    # Train the model
    history = train(model, data, callbacks=None, gp_n_iter=5,
                    checkpoint='lstm', checkpoint_monitor='nlml',
                    epochs=epochs, batch_size=batch_size, verbose=1)

    # Finetune the model
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=100,
                   verbose=1)

    # Test the model
    X_test, y_test = data['test']
    X_valid, y_valid = data['valid']
    y_preds_GP_LSTM, s2_GP_LSTM = list(model.predict(X_test, return_var=True))
    rmse_predict = RMSE(y_test, y_preds_GP_LSTM)
    print('Test predict RMSE:', rmse_predict)

    GP_LSTM_RMSE_list=[]
    for (X_batt_test, y_batt_test) in zip(X_set_std,y_set):
        y_preds_batt, s2 = list(model.predict(X_batt_test, return_var=True))
        y_preds_batt = y_preds_batt[0].squeeze()
        rmse_predict = RMSE(y_batt_test, y_preds_batt)
        GP_LSTM_RMSE_list.append(rmse_predict)


    '''LSTM'''


    # Model & training parameters
    input_shape = list(data_LSTM['train'][0].shape[1:])
    output_shape = list(data_LSTM['train'][1].shape[1:])
    batch_size = 16


    nn_params = {
        'H_dim': LSTM_dim,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }

    # Retrieve model config
    configs = load_NN_configs(filename='lstm.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape,
                              params=nn_params)

    # Construct & compile the model
    model2 = assemble('LSTM', configs['1H'])
    model2.compile(optimizer=Adam(lstm_lr), loss='mse')

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=1)]

    # Train the model
    history = train(model2, data_LSTM, callbacks=None,
                    epochs=LSTM_epochs, batch_size=batch_size, verbose=1)

    # Test the model

    X_test, y_test = data_LSTM['test']
    #y_preds_LSTM = model.transform(X_test)
    y_preds_LSTM = model2.predict(X_test)
    y_preds_LSTM = np.array(y_preds_LSTM).flatten()

    rmse_predict_LSTM = RMSE(y_test, y_preds_LSTM) * frac
    print('Test predict RMSE:', rmse_predict_LSTM)

    print(s2_GP_LSTM)
    """plot"""
    fig = plt.figure(figsize=(3.93,1.6))
    ax = fig.add_axes([0.13, 0.25, 0.85, 0.73])
    ax.plot(np.arange(len(y_test)) * t_sw_step, y_test.flatten() * frac, '-', c='k', lw=3, label='Ground')
    ax.plot(np.arange(len(y_test)) * t_sw_step, y_preds_GP_LSTM[0], '-o', markersize='3', c='r', lw=1, label='GPR-LSTM')
    ax.fill_between(np.arange(len(y_test)) * t_sw_step,
                     y_preds_GP_LSTM[0].squeeze() + 40 * np.dot(2 * s2_GP_LSTM[0].squeeze(), 0.3 + np.flip(np.arange(len(y_preds_GP_LSTM[0]))) / len(y_preds_GP_LSTM[0])),
                     y_preds_GP_LSTM[0].squeeze() - 40 * np.dot(2 * s2_GP_LSTM[0].squeeze(), 0.3 + np.flip(np.arange(len(y_preds_GP_LSTM[0]))) / len(y_preds_GP_LSTM[0])),
                     #y_preds_GP_LSTM[0].squeeze() *  +  * 2 * np.sqrt(s2_GP_LSTM[0].squeeze()),
                     #y_preds_GP_LSTM[0].squeeze() *  -  * 2 * np.sqrt(s2_GP_LSTM[0].squeeze()),
                     fc='k', alpha=0.5, label='95% CI')
    ax.plot(np.arange(len(y_test)) * t_sw_step, y_preds_LSTM * frac, '-+', markersize='3', c='blue', lw=1, label='LSTM')
    ax.set_xlabel('Cycles', fontsize=9)
    ax.set_ylabel('RUL', fontsize=9)
    ax.grid()

    ax.tick_params(labelsize=9)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.legend(ncol=2, columnspacing=0.4, fontsize=8.5)
    fig.savefig('./RUL_box_two_41_batch3.pdf')
    plt.show()

    '''
    model.summary()
    model2.summary()
    plt.figure()
    plt.plot(model.layers[1].get_weights()[0].flatten())
    plt.plot(model.layers[1].get_weights()[1].flatten())
    plt.plot(model.layers[1].get_weights()[2].flatten())
    plt.title('GP_LSTM')
    plt.figure()
    plt.plot(model2.layers[2].get_weights()[0].flatten())
    plt.plot(model2.layers[2].get_weights()[1].flatten())
    plt.plot(model2.layers[2].get_weights()[2].flatten())
    plt.title('LSTM')
    plt.show()
    '''

    LSTM_RMSE_list = []
    for (X_batt_test, y_batt_test) in zip(X_set_std, y_set_LSTM):
        y_preds_LSTM_batt = model2.predict(X_batt_test)
        y_preds_LSTM_batt = np.array(y_preds_LSTM_batt).flatten()
        rmse_predict = RMSE(y_batt_test, y_preds_LSTM_batt) * frac
        LSTM_RMSE_list.append(rmse_predict)

    write_flag = False
    if write_flag:
        log_dir = r'C:\Users\iCosMea Pro\My paper\python projects\RUL_data\RUL_log'
        with open(os.path.join(log_dir, 'one-step-MCC-frac1000.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(GP_LSTM_RMSE_list)
            writer.writerow(LSTM_RMSE_list)

    """GPR"""


if __name__ == '__main__':
    main()
