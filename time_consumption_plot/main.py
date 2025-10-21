import csv
import matplotlib.pyplot as plt
import re
import numpy as np
import os

plt.rc('font', family='Times New Roman')


def main():
    label = ['2', '4', '6', '8', '10']
    GP_LSTM_time = [482.76, 496.15, 549.62, 552.21, 608.46]
    GP_time = [76.747, 681.675, 1430.68, 2962.09, 7454.493]
    LSTM_time = [72.04, 141.54, 195.88, 266.3, 382.65]
    num_sample = [2080, 4575, 6580, 8853, 11263]

    fig = plt.figure(figsize=[3.93, 3.93])
    ax = fig.add_subplot()
    x_0 = np.array(list(range(len(label))))
    bar_width = 0.35
    move = bar_width / 2
    x_left = x_0 - move
    x_right = x_0 + move

    ax.bar(x_left, GP_LSTM_time, width=bar_width, label='GPR-LSTM')
    ax.bar(x_right, GP_time, width=bar_width, label='GPR')

    ax.set_xticks(x_0)
    ax.set_xticklabels(label)
    ax.set_xlabel('Training battery number')
    ax.set_ylabel('Time (s)')

    ax_samples = ax.twinx()
    ax_samples.plot(range(len(num_sample)), num_sample, 'o-', color='g', label='Sample number')
    ax_samples.set_ylabel('Sample number')
    '''
    hds = hd1 + hd2 + hd3
    labs = [l.get_label() for l in hds]
    ax.legend(hds, labs, loc=0)
    '''
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_samples.get_legend_handles_labels()


    ax.tick_params(labelsize=8)
    ax_samples.tick_params(labelsize=8)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

plt.show()
