import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import re
import os
from scipy.stats import pearsonr, spearmanr
np.set_printoptions(threshold=np.inf)

plt.rc('font', family='Times New Roman')


def label_get():
    path_1 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_1_summary'
    path_2 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_2_summary'
    path_3 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_3_summary'
    batch_1_list = os.listdir(path_1)
    batch_1_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_2_list = os.listdir(path_2)
    batch_2_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_3_list = os.listdir(path_3)
    batch_3_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))

    label_1, label_2, label_3 = [], [], []
    for ba in batch_1_list:
        Qd_1_ba = []
        with open(os.path.join(path_1, ba), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            first_zero = next(reader)
            for row in reader:
                Qd_1_ba.append(float(row[1]) / 1.1)
        label_1.append(Qd_1_ba)

    for ba in batch_2_list:
        Qd_2_ba = []
        with open(os.path.join(path_2, ba), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            first_zero = next(reader)
            for row in reader:
                Qd_2_ba.append(float(row[1]) / 1.1)
        label_2.append(Qd_2_ba)

    for ba in batch_3_list:
        Qd_3_ba = []
        with open(os.path.join(path_3, ba), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            first_zero = next(reader)
            for row in reader:
                Qd_3_ba.append(float(row[1]) / 1.1)
        label_3.append(Qd_3_ba)

    return label_1, label_2, label_3


def corralation_analysis(label_1, label_2, label_3, feature_label_flag=False):
    feature_path_1 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_1_feature'
    feature_path_2 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_2_feature'
    feature_path_3 = r'C:\Users\iCosMea Pro\My paper\python projects\data_process\batch_3_feature'

    feature_batch_1_list = os.listdir(feature_path_1)
    feature_batch_1_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    feature_batch_2_list = os.listdir(feature_path_2)
    feature_batch_2_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    feature_batch_3_list = os.listdir(feature_path_3)
    feature_batch_3_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))

    batch_1_feature = []
    for ba in feature_batch_1_list:
        with open(os.path.join(feature_path_1, ba), 'r') as f:
            ba_feature = []
            reader = csv.reader(f)
            for row in reader:
                cy_feature = []
                for item in row:
                    cy_feature.append(float(item))
                ba_feature.append(cy_feature)
            batch_1_feature.append(list(ba_feature))

    batch_2_feature = []
    for ba in feature_batch_2_list:
        with open(os.path.join(feature_path_2, ba), 'r') as f:
            ba_feature = []
            reader = csv.reader(f)
            for row in reader:
                cy_feature = []
                for item in row:
                    cy_feature.append(float(item))
                ba_feature.append(cy_feature)
            batch_2_feature.append(list(ba_feature))

    batch_3_feature = []
    for ba in feature_batch_3_list:
        with open(os.path.join(feature_path_3, ba), 'r') as f:
            ba_feature = []
            reader = csv.reader(f)
            for row in reader:
                cy_feature = []
                for item in row:
                    cy_feature.append(float(item))
                ba_feature.append(cy_feature)
            batch_3_feature.append(list(ba_feature))


    '''清除SOH低于80%'''
    batch_1_feature_new, label_1_new = [], []
    for ba_feature_raw, ba_label_raw in zip(batch_1_feature, label_1):
        ba_feature = [i for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        ba_label = [j for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        batch_1_feature_new.append(list(ba_feature))
        label_1_new.append(list(ba_label))

    batch_2_feature_new, label_2_new = [], []
    for ba_feature_raw, ba_label_raw in zip(batch_2_feature, label_2):
        ba_feature = [i for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        ba_label = [j for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        batch_2_feature_new.append(list(ba_feature))
        label_2_new.append(list(ba_label))

    batch_3_feature_new, label_3_new = [], []
    for ba_feature_raw, ba_label_raw in zip(batch_3_feature, label_3):
        ba_feature = [i for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        ba_label = [j for i, j in zip(ba_feature_raw, ba_label_raw) if j > 0.805]
        batch_3_feature_new.append(list(ba_feature))
        label_3_new.append(list(ba_label))

    '''rename'''
    batch_1_feature, batch_2_feature, batch_3_feature, label_1, label_2, label_3 =\
        feature_norm(batch_1_feature_new[0:]), feature_norm(batch_2_feature_new[0:]), feature_norm(batch_3_feature_new[0:]), label_1_new, label_2_new, label_3_new

    if feature_label_flag:
        feature_all = batch_1_feature + batch_2_feature + batch_3_feature
        #feature_all = feature_all[10:11]
        label_all = label_1 + label_2 +label_3
        #label_all = label_all[10:11]

        two_stage = [0, 1, 2, 3, 4, 5, 6, 7, 53, 54, 55, 61, 62, 70, 71, 72, 101, 104, 110, 117, 126, 136, 139]

        three_stage = [i for i in range(140) if i not in two_stage]

        '''feature_ba[-1][6] < 28.68 and np.nanmin(label_ba) < 0.81'''
        curve_bias = [46, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
        curve_bi = [0, 1, 2, 3, 4, 8, 10, 12, 13, 22, 35, 39, 70, 71, 72, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]
        """feature_ba[-1][6] < 28.78 and np.nanmin(label_ba) < 0.81"""
        curve_batch_1 = range(0, 46)
        curve_batch_2 = range(46, 94)
        curve_batch_3 = range(94, 140)

        """删除异常部分,且只选取目标特征"""
        feature_new, label_all_new = [], []
        for feature_ba, label_ba in zip(feature_all, label_all):
            for feature in [6]:
                x = np.array(feature_ba).T[feature]
                y = label_ba

                x_new, y_new = [], []
                for index, (x_i, y_i) in enumerate(zip(x, y)):
                    if index == 0:
                        x_last = x_i
                        x_new.append(x_i)
                        y_new.append(y_i)
                        continue
                    if abs(x_i - x_last) < 0.2:
                        y_new.append(y_i)
                        x_new.append(x_i)
                        x_last = x_i
            feature_new.append(x_new)
            label_all_new.append(y_new)
        """分批次描述特征7与SOH的关系"""
        """
        for index, (feature_ba, label_ba) in enumerate(zip(feature_new, label_all_new)):
            # for feature in range(0,14):
            x = feature_ba
            y = label_ba
            if len(x) < 2:
                continue
            print('len', len(x), len(y))
            # if index in two_stage:
            if index in curve_batch_1:
                if index == 0:
                    plt.scatter(x, y, color='g', s=5, alpha=0.4, label='batch_1')
                else:
                    plt.scatter(x, y, color='g', s=5, alpha=0.4)
                    #plt.plot(x, y, color='g', linewidth=2, alpha=0.6)
            elif index in curve_batch_2:
                if index == 46:
                    plt.scatter(x, y, color='y', s=5, alpha=0.4, label='batch_2')
                else:
                    plt.scatter(x, y, color='y', s=5, alpha=0.4)
                    #plt.plot(x, y, linewidth=2, alpha=0.6)
            elif index in curve_batch_3:
                if index == 94:
                    plt.scatter(x, y, color='c', s=5, alpha=0.4, label='batch_3')
                else:
                    plt.scatter(x, y, color='c', s=5, alpha=0.4)
                    #plt.plot(x, y, linewidth=2, alpha=0.6)
            nan_index = np.argwhere(np.isnan(x))
            x = np.delete(x, nan_index)
            y = np.delete(y, nan_index)
            c, _ = pearsonr(x, y)
            plt.legend()
            plt.title(f'{6}, {c}')
        plt.show()
        """

        '''
        for index, (feature_ba, label_ba) in enumerate(zip(feature_all, label_all)):
            #for feature in range(0,14):
            for feature in [6]:
                x = np.array(feature_ba).T[feature]
                y = label_ba
                #if index in two_stage:
                if index in curve_batch_1:
                    if index == 0:
                        plt.scatter(x, y, color='g', s=5, alpha=0.4, label='batch_1')
                    else:
                        #plt.scatter(x, y, color='g', s=5, alpha=0.4)
                        plt.plot(x, y, color='g', linewidth=2, alpha=0.6)
                elif index in curve_batch_2:
                    if index == 46:
                        plt.scatter(x, y, color='y', s=5, alpha=0.4, label='batch_2')
                    else:
                        #plt.scatter(x, y, color='y', s=5, alpha=0.4)
                        plt.plot(x, y, linewidth=2, alpha=0.6)
                elif index in curve_batch_3:
                    if index == 94:
                        plt.scatter(x, y, color='c', s=5, alpha=0.4, label='batch_3')
                    else:
                        #plt.scatter(x, y, color='c', s=5, alpha=0.4)
                        plt.plot(x, y, linewidth=2, alpha=0.6)
                nan_index = np.argwhere(np.isnan(x))
                x = np.delete(x, nan_index)
                y = np.delete(y, nan_index)
                c, _ = pearsonr(x, y)
                plt.legend()
                plt.title(f'{feature}, {c}')
        plt.show()
        '''
    two_stage_batch_1 = [0, 1, 2, 3, 4, 5, 6, 7, 20, 21]
    two_stage_batch_2 = [7, 8, 9, 15, 16, 24, 25, 26]  # 46+
    two_stage_batch_3 = [7, 10, 16, 23, 32, 37, 42, 45]  # 94+

    '''batch_1_cor'''
    batch_1_cors_two = []
    batch_1_cors_s_two = []
    batch_1_cors_three = []
    batch_1_cors_s_three = []
    for index_cell, (ba_feature, ba_label) in enumerate(zip(batch_1_feature, label_1)):
        ba_cor = []
        ba_cor_s = []
        ba_feature = np.array(ba_feature).T
        ba_label = np.array(ba_label).T
        del_index = np.argwhere(np.isnan(ba_feature[0]))        # 删除nan对相关度分析的影响
        ba_feature = np.delete(ba_feature, del_index, axis=1)
        ba_label = np.delete(ba_label, del_index, axis=0)
        for index, each_feature in enumerate(ba_feature):
            if len(each_feature.tolist()) < 1:
                cor_each_feature = np.nan
                cor_each_feature_s = np.nan
            else:
                cor_each_feature, _ = pearsonr(each_feature, ba_label.squeeze())
                cor_each_feature_s, _ = spearmanr(each_feature, ba_label.squeeze())
            ba_cor.append(cor_each_feature)
            ba_cor_s.append(cor_each_feature_s)
        if index_cell in two_stage_batch_1:
            batch_1_cors_two.append(ba_cor)
            batch_1_cors_s_two.append(ba_cor_s)
        else:
            batch_1_cors_three.append(ba_cor)
            batch_1_cors_s_three.append(ba_cor_s)
    '''batch_2_cor'''
    batch_2_cors_two = []
    batch_2_cors_s_two = []
    batch_2_cors_three = []
    batch_2_cors_s_three = []
    for index_cell, (ba_feature, ba_label) in enumerate(zip(batch_2_feature, label_2)):
        ba_cor = []
        ba_cor_s = []
        ba_feature = np.array(ba_feature).T
        ba_label = np.array(ba_label).T
        del_index = np.argwhere(np.isnan(ba_feature[0]))  # 删除nan对相关度分析的影响
        ba_feature = np.delete(ba_feature, del_index, axis=1)
        ba_label = np.delete(ba_label, del_index, axis=0)
        for index, each_feature in enumerate(ba_feature):
            if len(each_feature.tolist()) < 1:
                cor_each_feature = np.nan
                cor_each_feature_s = np.nan
            else:
                cor_each_feature, _ = pearsonr(each_feature, ba_label.squeeze())
                cor_each_feature_s, _ = spearmanr(each_feature, ba_label.squeeze())
            ba_cor.append(cor_each_feature)
            ba_cor_s.append(cor_each_feature_s)
        if index_cell in two_stage_batch_2:
            batch_2_cors_two.append(ba_cor)
            batch_2_cors_s_two.append(ba_cor_s)
        else:
            batch_2_cors_three.append(ba_cor)
            batch_2_cors_s_three.append(ba_cor_s)
    '''batch_3_cor'''
    batch_3_cors_two = []
    batch_3_cors_s_two = []
    batch_3_cors_three = []
    batch_3_cors_s_three = []
    for index_cell, (ba_feature, ba_label) in enumerate(zip(batch_3_feature, label_3)):
        ba_cor = []
        ba_cor_s = []
        ba_feature = np.array(ba_feature).T
        ba_label = np.array(ba_label).T
        del_index = np.argwhere(np.isnan(ba_feature[0]))  # 删除nan对相关度分析的影响
        ba_feature = np.delete(ba_feature, del_index, axis=1)
        ba_label = np.delete(ba_label, del_index, axis=0)
        for index, each_feature in enumerate(ba_feature):
            if len(each_feature.tolist()) < 2:
                cor_each_feature = np.nan
                cor_each_feature_s = np.nan
            else:
                cor_each_feature, _ = pearsonr(each_feature, ba_label.squeeze())
                cor_each_feature_s, _ = spearmanr(each_feature, ba_label.squeeze())
            ba_cor.append(cor_each_feature)
            ba_cor_s.append(cor_each_feature_s)
        if index_cell in two_stage_batch_3:
            batch_3_cors_two.append(ba_cor)
            batch_3_cors_s_two.append(ba_cor_s)
        else:
            batch_3_cors_three.append(ba_cor)
            batch_3_cors_s_three.append(ba_cor_s)
    all_cors_two = np.concatenate([np.array(batch_1_cors_two), np.array(batch_2_cors_two), np.array(batch_3_cors_two)], axis=0)
    all_cors_two = all_cors_two[~np.isnan(all_cors_two).any(axis=1)]
    all_cors_three = np.concatenate([np.array(batch_1_cors_three), np.array(batch_2_cors_three), np.array(batch_3_cors_three)], axis=0)
    all_cors_three = all_cors_three[~np.isnan(all_cors_three).any(axis=1)]

    all_cors_s_two = np.concatenate([np.array(batch_1_cors_s_two), np.array(batch_2_cors_s_two), np.array(batch_3_cors_s_two)], axis=0)
    all_cors_s_two = all_cors_s_two[~np.isnan(all_cors_s_two).any(axis=1)]
    all_cors_s_three = np.concatenate([np.array(batch_1_cors_s_three), np.array(batch_2_cors_s_three), np.array(batch_3_cors_s_three)], axis=0)
    all_cors_s_three = all_cors_s_three[~np.isnan(all_cors_s_three).any(axis=1)]
    return np.abs(all_cors_two), np.abs(all_cors_s_two), np.abs(all_cors_three), np.abs(all_cors_s_three)


def feature_norm(feature):
    """
    会降低计算速度
    :param feature:
    :return:
    """

    feature_rt = []
    for ba in feature:
        ba_rt = []
        ba = np.array(ba).T.tolist()     # 先分特征再分周期
        for index, fea in enumerate(ba):
            fea_rt = []
            if index == 6:
                for fea_val in fea:
                    # item = (fea_val - np.nanmean(fea)) / np.nanstd(fea)
                    #item = (fea_val - np.nanmin(fea)) / (np.nanmax(fea) - np.nanmin(fea))
                    item = fea_val
                    fea_rt.append(item)
                ba_rt.append(fea_rt)
            else:
                ba_rt.append(fea)
        ba_rt = np.array(ba_rt).T.tolist()  # 返回先分周期再分特征
        feature_rt.append(ba_rt)

    return feature_rt


def cor_plot(all_cors, all_cors_s, filename):
    """all_cors: Pearson correlation
    all_cors_s: Spearman's rank correlation"""
    feature_labels = [r'$\overline {D}_1$', r'$\overline {D}_2$', r'$\overline {D}_3$', r'$\overline {D}_4$',
                      r'$\overline {D}_5$', r'$\overline {D}_6$', r'$\overline {A}_6$', r'Var${(D_1)}$',
                      r'Var${(D_2)}$', r'Var${(D_3)}$', r'Var${(D_4)}$', r'Var${(D_5)}$', r'Var${(D_6)}$', r'Var${(A_6)}$']
    fig_p = plt.figure(figsize=[3.94, 2.2])
    ax0 = fig_p.add_axes(rect=[0.07,0.14,0.9,0.85])
    ax0.boxplot(all_cors, labels=feature_labels, boxprops=dict(linewidth=1, color='blue'),
                flierprops=dict(marker='+', markeredgecolor='red'), medianprops=dict(linewidth=2, c='tab:red'),
                whiskerprops=dict(linestyle='--'))
    ax0.tick_params(axis='x', labelsize=8, rotation=30, pad=-1)
    ax0.tick_params(axis='y', labelsize=8)
    ax0.spines['top'].set_linewidth(1.5)
    ax0.spines['bottom'].set_linewidth(1.5)
    ax0.spines['left'].set_linewidth(1.5)
    ax0.spines['right'].set_linewidth(1.5)

    # savefig
    #fig_p.savefig(filename + '_spearman.pdf')
    #fig_p.savefig(filename + '_spearman.eps')
    # fig_p.savefig(filename+'.pdf')
    # fig_p.savefig(filename + '.eps')
    '''
    plt.figure()
    plt.boxplot(all_cors_s)
    plt.title('spearman')
    '''
    plt.show()


def main():
    #two_stage = [0 , 1, 2, 3, 4, 53, 54, 55, 61, 62, 70, 71, 72, 101, 104, 110, 117, 126, 136, 139]
    #three_stage = [i for i in range(140) if i not in two_stage]

    label_1, label_2, label_3 = label_get()
    all_cors_two, all_cors_p_two, all_cors_three, all_cors_p_three = corralation_analysis(label_1, label_2, label_3, True)
    all_cors_two = np.delete(all_cors_two, list(range(0,0)) + list(range(len(all_cors_two),len(all_cors_two))), axis=0)
    all_cors_p_two = np.delete(all_cors_p_two, list(range(0, 0)) + list(range(len(all_cors_p_two), len(all_cors_p_two))), axis=0)
    all_cors_three = np.delete(all_cors_three, list(range(0, 0)) + list(range(len(all_cors_three), len(all_cors_three))),
                             axis=0)
    all_cors_p_three = np.delete(all_cors_p_three,
                               list(range(0, 0)) + list(range(len(all_cors_p_three), len(all_cors_p_three))), axis=0)

    print('Pearson one-step:', all_cors_two)
    print('Pearson two-step:', all_cors_three)
    print('Spearman one-step:', all_cors_p_two)
    print('Spearman one-step:', all_cors_p_three)

    print('Pearson one-step Avg.:', np.round(np.mean(all_cors_two, axis=0),3))
    print('Pearson two-step Avg.:', np.round(np.mean(all_cors_three, axis=0),3))
    print('Spearman one-step Avg.:', np.round(np.mean(all_cors_p_two, axis=0),3))
    print('Spearman one-step Avg.:', np.round(np.mean(all_cors_p_three, axis=0),3))

    cor_plot(all_cors_two, all_cors_p_two, filename='two stage feature relation')
    cor_plot(all_cors_three, all_cors_p_three, filename='three stage feature relation')


if __name__ == '__main__':
    main()