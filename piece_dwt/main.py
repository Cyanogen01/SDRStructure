import pywt
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re
from shutil import copyfile
from scipy.stats import pearsonr
import matplotlib as mpl

plt.rc('font', family='Times New Roman')


def dwt_cycle(cycle_chosen):
    with open(cycle_chosen, 'r') as f:
        csv_reader = csv.reader(f)
        t, V = [], []
        for row in csv_reader:
            t.append(float(row[0]))
            V.append(float(row[1]))

    if len(t) < 10:  # 排除异常
        return False, False
    # 假设t均匀
    wavelet_name = 'haar'
    padding_mode = 'constant'
    level = pywt.dwt_max_level(len(V), wavelet_name)
    wavelet_obj = pywt.Wavelet(wavelet_name)

    cA_last = V
    cD_layer = []
    for index, layer in enumerate(range(level), start=0):
        (cA, cD) = pywt.dwt(cA_last, wavelet_name, mode=padding_mode)
        cD_layer.append(cD)
        cA_last = cA.copy()

    return cD_layer, cA_last


def dwt_plot(cD_layer_ba, cA_last_ba):
    level = 6
    ax = []
    fig = plt.figure(figsize=[1.96, 3.15])
    c_max = len(cD_layer_ba)
    cycle_count = 0

    for cD_layer, cA_last in zip(cD_layer_ba, cA_last_ba):
        # cD_layer = cD_layer[0]      # 除去为了list np.nan的list
        # cA_last = cA_last[0]
        if True in np.isnan(cD_layer[0]):
            continue
        rgb = plt.cm.viridis(cycle_count / c_max)
        # rgb = [0.2 + (cycle_count / c_max) ** 5 * 0.45, 0.9 - (cycle_count / c_max) ** 5 * 0.8, 0.3]       # green to red 5次曲线提高绿色成分
        for index, layer in enumerate(range(level)):
            # ax.append(fig.add_subplot(level + 1, 1, index + 1))
            ax.append(plt.axes([0.15, 0.8 - index * 0.12, 0.69, 0.1]))
            ax[index].plot(range(len(cD_layer[index])), abs(np.array(cD_layer[index])), color=rgb, alpha=0.5, linewidth=1)
            ax[index].get_xaxis().set_visible(False)
            ax[index].tick_params(labelsize=8)
            ax[index].spines['top'].set_linewidth(1.5)
            ax[index].spines['bottom'].set_linewidth(1.5)
            ax[index].spines['left'].set_linewidth(1.5)
            ax[index].spines['right'].set_linewidth(1.5)

        # ax.append(fig.add_subplot(level + 1, 1, level + 1))
        ax.append(plt.axes([0.15, 0.8 - 6 * 0.12, 0.69, 0.1]))
        ax[-1].plot(range(len(cA_last)), abs(np.array(cA_last)), color=rgb, alpha=0.5, linewidth=0.2)
        ax[-1].get_xaxis().set_visible(False)
        ax[-1].tick_params(labelsize=8)
        ax[-1].spines['top'].set_linewidth(1.5)
        ax[-1].spines['bottom'].set_linewidth(1.5)
        ax[-1].spines['left'].set_linewidth(1.5)
        ax[-1].spines['right'].set_linewidth(1.5)
        cycle_count = cycle_count + 1

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=c_max, vmax=0), cmap=plt.cm.viridis),
                      ax=(ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6]), fraction=0.1)
    cb.ax.set_title('Cycles', size=8)
    cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(hspace=0.25, right=0.8)
    name = 'One-step MCC'
    fig.suptitle(name, x=0.45, y=0.95, fontsize=9)
    fig.savefig(name+'.png')
    fig.savefig(name + '.eps')

def feature_extract(cD_layer_ba, cA_last_ba):
    """输出14个特征：6层细节层和1层大约层的平均值和方差"""
    ba_feature = np.empty((1, 14))
    for cD_layer_cy, cA_last_cy in zip(cD_layer_ba, cA_last_ba):
        layer_mean = np.array([])
        layer_std = np.array([])
        for cD_layer in cD_layer_cy:
            layer_mean = np.concatenate([layer_mean, np.mean(cD_layer)], axis=None)
            layer_std = np.concatenate([layer_std, np.std(cD_layer)], axis=None)
        layer_mean = np.concatenate([layer_mean, np.mean(cA_last_cy)], axis=None)
        layer_std = np.concatenate([layer_std, np.std(cA_last_cy)], axis=None)

        layer_feature = np.concatenate([layer_mean, layer_std], axis=None)

        if len(layer_feature) != 14:  # 1.对nan情况 2.对空文件情况
            layer_feature = np.array([np.nan for _ in range(14)])  # 标准化格式
        ba_feature = np.concatenate([ba_feature, layer_feature.reshape(1, layer_feature.shape[0])], axis=0)
    ba_feature = np.delete(ba_feature, 0, axis=0)  # 除去初始化产生的行
    return ba_feature


def feature_write(batch_list, batch_partial, batch_feature_dir, write_flag=False):
    for index, ba in enumerate(batch_list[0:]):  # 命名从1开始的
        print(ba)
        cD_layer_ba, cA_last_ba = [], []
        cy_list = os.listdir(os.path.join(batch_partial, ba))
        cy_list.sort(key=lambda l: int(re.findall('\d+', l)[2]))
        for jndex, cycle in enumerate(cy_list):
            if jndex % 1 == 0:
                cycle_chosen = os.path.join(os.path.join(batch_partial, ba), cycle)  # level = 6
                cD_layer_cycle, cA_last_cycle = dwt_cycle(cycle_chosen)
                if cD_layer_cycle == False:
                    cD_layer_cycle = [np.array([np.nan, np.nan])]  # 适配np.isnan
                    cA_last_cycle = [np.array([np.nan, np.nan])]
                cD_layer_ba.append(cD_layer_cycle)
                cA_last_ba.append(cA_last_cycle)

        if write_flag:
            ba_feature = feature_extract(cD_layer_ba, cA_last_ba)
            with open(os.path.join(batch_feature_dir, 'batch_' + re.findall('\d+', str(batch_list))[0] + '_cell_' + str(
                    index + 1) + '_feature.csv'), 'w',
                      newline='') as f:
                writer = csv.writer(f)
                for row in ba_feature:
                    writer.writerow(row)

        if not write_flag:
            dwt_plot(cD_layer_ba, cA_last_ba)
            plt.show()


def main():
    root_dir = r'G:\My paper\python projects\data_process'
    batch_1_partial = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_1_partial'
    batch_2_partial = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_2_partial'
    batch_3_partial = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_3_partial'
    batch_1_list = os.listdir(batch_1_partial)
    batch_2_list = os.listdir(batch_2_partial)
    batch_3_list = os.listdir(batch_3_partial)
    batch_1_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_2_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_3_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))

    batch_1_feature_dir = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_1_feature'
    batch_2_feature_dir = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_2_feature'
    batch_3_feature_dir = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_3_feature'

    write_flag = False
    '''for batch_list, batch_partial, batch_feature_dir in zip([batch_1_list, batch_2_list, batch_3_list],
                                                            [batch_1_partial, batch_2_partial, batch_3_partial],
                                                            [batch_1_feature_dir, batch_2_feature_dir, batch_3_feature_dir]):'''
    batch_list, batch_partial, batch_feature_dir = [batch_1_list[3]], batch_1_partial, batch_1_feature_dir
    # batch_list, batch_partial, batch_feature_dir = [batch_1_list[39]], batch_1_partial, batch_1_feature_dir
    feature_write(batch_list, batch_partial, batch_feature_dir, write_flag)


if __name__ == '__main__':
    main()
