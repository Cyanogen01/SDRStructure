import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import re
import os


def partial_curve(cycledir, writeflag=False, *arg):
    cycle_list = os.listdir(cycledir)
    cycle_list_clean = [i for i in cycle_list if '_lin' not in i]   # 除去lin文件
    cycle_list_clean.sort(key=lambda l: int(re.findall('\d+', l)[2]))   # 排序

    cycle_start = 0

    cycle_list_clean = cycle_list_clean[cycle_start:]

    for ba in cycle_list_clean:
        with open (os.path.join(cycledir, ba), 'r') as f:
            t, I, V = [], [], []
            csvreader = csv.reader(f)
            header = next(csvreader)
            for row in csvreader:
                t.append(float(row[1]))
                I.append(float(row[3]))
                V.append(float(row[4]))
        print(os.path.join(cycledir, ba))
        #t_partial, V_partial = partial_cut_old(t, I, V)
        flag = False
        if int(re.findall('\d+',ba)[1]) == 4:
            if int(re.findall('\d+', ba)[2]) > 940:
                flag = not writeflag
        t_partial, V_partial = partial_cut_dV(t, I, V, flag)
        cycle_num = int(re.findall('\d+', ba)[2])

        if not writeflag:
            c_max = len(cycle_list_clean)
            c_min = cycle_start
            #rgb = plt.cm.viridis((cycle_num - c_min) / (c_max - c_min))
            rgb = [0.2 + (cycle_num / c_max) ** 4 * 0.45, 0.9 - (cycle_num / c_max) ** 4 * 0.8, 0.3]  # green to red 4次曲线提高绿色成分
            plt.plot(t_partial, V_partial, color=rgb, alpha=0.4)
            #plt.plot(range(len(V_partial)), V_partial, color=rgb, alpha=0.4)
            plt.grid(True)
            #plt.xlim([t_partial[0], t_partial[-1]])
            plt.xlabel('time / min')
            plt.ylabel('voltage / V')
            #plt.title('partial charging profiles')
        if writeflag:
            arg[0].append(t_partial.tolist())
            arg[1].append(V_partial.tolist())


def partial_cut_dV(t, I, V, flag=False):
    """
    通过电压差分确定
    :param t:
    :param I:
    :param V:
    :param flag:
    :return:
    """
    if t[-1] > 100:
        print('time error')
        return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error

    V_last = I[0]
    epsilon = 1e-6  # 避免开头电流连续0产生的除零错误
    Vd = []
    for index, (t_now, V_now, I_now) in enumerate(zip(t, V, I)):
        if index == 0:
            Vd.append(0)
            V_last = V_now
            continue
        if I_now > -0.01 and V_now - V_last < 1.5 and V_now > 3.0:  # 判断是否在充电阶段，排除放电阶段零电流点突变，排除初始阶段
            Vd.append(V_now - V_last)
            V_last = V_now
        else:
            Vd.append(0)
            V_last = V_now

    if flag:
        plt.close()
        plt.plot(t, Vd)
        plt.show()
    start_index = np.argmax(np.array(Vd)) # numpy的逻辑与运算
    print(start_index)
    '''the following follow as the old'''
    index_switch = np.arange(start_index - 100, start_index + 100)
    index_switch_delta = index_switch - 1
    index_switch = [int(i) for i in index_switch]
    index_switch_delta = [int(i) for i in index_switch_delta]

    time_switch = np.array(t)[index_switch]
    time_switch_delta = np.array(t)[index_switch_delta]

    index_zrm_raw = [i for i, t, t_last in zip(
        index_switch, time_switch,
        time_switch_delta) if t - t_last > 1e-2 or i == start_index]
    index_zrm_raw = np.array(index_zrm_raw)

    zrm_start = int(np.argwhere(np.abs(index_zrm_raw - start_index) < 1e-6))
    print('zrm_start', zrm_start)
    print('len',len(index_zrm_raw))
    #zrm_start = index_zrm_raw[np.argwhere(index_zrm_raw) == index_zrm_raw[0]]
    '''一共65个有效点，转换末尾前30个点开始'''
    zrm_index = index_zrm_raw[zrm_start - 30: zrm_start + 35]
    zrm_t = np.array(t)[zrm_index]
    zrm_I = np.array(I)[zrm_index]
    zrm_V = np.array(V)[zrm_index]

    return zrm_t, zrm_V



def partial_cut_fd(t, I, V, flag=False):
    """
    forward difference
    通过前向差分找到突变点，以进行多阶段分割
    """
    '''
    t0 = t[0]
    for tt in t:
        if tt - t0 > 0.8:
            print('time interval error')
            return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error
        t0 = tt
    '''
    if t[-1] > 100:
        print('time error')
        return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error

    t_last = t[0]
    I_last = I[0]
    epsilon = 1e-6  # 避免开头电流连续0产生的除零错误
    fd_It = []
    for index, (t_now, I_now) in enumerate(zip(t, I)):
        if index == 0:
            fd_It.append(0)
            t_last = t_now
            I_last = I_now
            continue
        if I_now > -0.01:   # 判断是否在充电阶段
            '''if abs(I_now - I_last) > 0.8:
                fd_It.append((I_now - I_last) * 1e4)
            else:
                fd_It.append((I_now - I_last) / (t_now - t_last + epsilon))'''
            fd_It.append(I_now - I_last)
            t_last = t_now
            I_last = I_now
        else:
            fd_It.append(0)
            t_last = t_now
            I_last = I_now

    if flag:
        plt.close()
        plt.plot(t, fd_It)
        plt.show()
    start_index = np.argwhere(np.logical_and((np.abs(np.array(fd_It)) > 0.6), (np.abs(fd_It) < 10)))   # numpy的逻辑与运算
    if(len(np.argwhere(start_index > 1000)) > 0):
        start_index = np.delete(start_index, np.argwhere(start_index > 1000).flatten()[0], axis=0)    # 大于这个周期的转换被除去
    print(start_index)
    if len(start_index.flatten()) < 2:
        print('can not find abrupt')
        return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error
    start_index = start_index[-2]     # 倒数第2个代表MCC到CCCV的转换点

    if start_index > 650:   # 大于这个周期的转换被认为是错误 已经弃用这个错误判断
        print('neglected')
        return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error

    print('start_index', start_index)

    '''the following follow as the old'''
    index_switch = np.arange(start_index - 100, start_index + 100)
    index_switch_delta = index_switch - 1
    index_switch = [int(i) for i in index_switch]
    index_switch_delta = [int(i) for i in index_switch_delta]

    time_switch = np.array(t)[index_switch]
    time_switch_delta = np.array(t)[index_switch_delta]

    index_zrm_raw = [i for i, t, t_last in zip(
        index_switch, time_switch,
        time_switch_delta) if t - t_last > 1e-2 or i == start_index]
    index_zrm_raw = np.array(index_zrm_raw)

    zrm_start = int(np.argwhere(np.abs(index_zrm_raw - start_index) < 1e-6))
    print('zrm_start', zrm_start)
    print('len',len(index_zrm_raw))
    #zrm_start = index_zrm_raw[np.argwhere(index_zrm_raw) == index_zrm_raw[0]]
    '''一共65个有效点，转换前第20个点开始'''
    zrm_index = index_zrm_raw[zrm_start - 20: zrm_start + 45]
    zrm_t = np.array(t)[zrm_index]
    zrm_I = np.array(I)[zrm_index]
    zrm_V = np.array(V)[zrm_index]
    if np.any(np.array(zrm_V) > 3.6):
        return np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])  # return error

    return zrm_t, zrm_V


def partial_cut_old(t, I, V):
    """
    old:通过切换到CCCV的0时间段电流确定
    找到最大连续区间
    除去异常双采样点
    """
    t0 = t[0]
    for tt in t:
        if tt - t0 > 0.8:
            return np.array([0,0,0,0,0]), np.array([0,0,0,0,0])     # return error
        t0 = tt

    zero_piece = [i for i, n in enumerate(I) if abs(n) < 1e-3 and i != 0]
    zero_piece_fill = []
    for i in range(zero_piece[0], zero_piece[-1]):

        if (i - 1 in zero_piece) or (i + 1 in zero_piece) or (i in zero_piece):     # 连接切换过程中的突变情况
            zero_piece_fill.append(i)
    zero_piece_fill = zero_piece_fill[1:]   # 除去第一个多余的
    zero_piece_fill = [i for i in zero_piece_fill if i < 900] # 排除后面过多0电流
    con_list = []
    for i, item in enumerate(zero_piece_fill):
        if i == 0:
            last = item
            con = 0
            con_list.append(con)
            continue
        if item == last + 1:
            con = con + 1
            last = item
            con_list.append(con)
        else:
            con = 0
            last = item
            con_list.append(con)

    if len(zero_piece_fill) < 5:
        return np.array([0,0,0,0,0]), np.array([0,0,0,0,0])     # return error
    start_index = zero_piece_fill[np.argmax(con_list) - np.max(con_list)]

    """粗取199个"""
    '''一共65个有效点，转换前第20个点开始'''

    index_switch = np.arange(start_index-100, start_index+100)
    index_switch_delta = index_switch - 1
    index_switch = [int(i) for i in index_switch]
    index_switch_delta = [int(i) for i in index_switch_delta]

    time_switch = np.array(t)[index_switch]
    time_switch_delta = np.array(t)[index_switch_delta]

    index_zrm_raw = [i for i, t, t_last in zip(
        index_switch, time_switch,
        time_switch_delta) if t - t_last > 1e-2 or i == start_index]
    index_zrm_raw = np.array(index_zrm_raw)
    zrm_start = int(np.argwhere(np.abs(index_zrm_raw - start_index) < 1e-6))
    zrm_index = index_zrm_raw[zrm_start - 20: zrm_start + 35]
    zrm_t = np.array(t)[zrm_index]
    zrm_I = np.array(I)[zrm_index]
    zrm_V = np.array(V)[zrm_index]
    return zrm_t, zrm_V


def csv_piece_write(t_all_partial, V_all_partial):
    root_dir = r'G:\My paper\python projects\data_process'
    batch_1_partial, batch_2_partial, batch_3_partial = zip(t_all_partial, V_all_partial)
    batch_1_t, batch_1_V = batch_1_partial[0], batch_1_partial[1]
    batch_2_t, batch_2_V = batch_2_partial[0], batch_2_partial[1]
    batch_3_t, batch_3_V = batch_3_partial[0], batch_3_partial[1]

    """batch1"""
    index = 0
    os.mkdir(os.path.join(root_dir, 'batch_1_partial'))
    for ba_t, ba_V in zip(batch_1_t, batch_1_V):
        ba_dir = os.path.join(os.path.join(root_dir, 'batch_1_partial'), ('batch_1_cell_' + str(index + 1)))
        os.mkdir(ba_dir)
        indexx = 0
        for t_cy, V_cy in zip(ba_t, ba_V):
            with open(os.path.join(ba_dir, ('batch_1_cell_' + str(index + 1) + '_cycle_' + str(indexx + 2)) + '.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                for t_, V_ in zip(t_cy, V_cy):
                    writer.writerow([t_, V_])
            indexx = indexx + 1
        index = index + 1

    """batch2"""
    index = 0
    os.mkdir(os.path.join(root_dir, 'batch_2_partial'))
    for ba_t, ba_V in zip(batch_2_t, batch_2_V):
        ba_dir = os.path.join(os.path.join(root_dir, 'batch_2_partial'), ('batch_2_cell_' + str(index + 1)))
        os.mkdir(ba_dir)
        indexx = 0
        for t_cy, V_cy in zip(ba_t, ba_V):
            with open(os.path.join(ba_dir, ('batch_2_cell_' + str(index + 1) + '_cycle_' + str(indexx + 2)) + '.csv'),
                      'w', newline='') as f:
                writer = csv.writer(f)
                for t_, V_ in zip(t_cy, V_cy):
                    writer.writerow([t_, V_])
            indexx = indexx + 1
        index = index + 1

    """batch3"""
    index = 0
    os.mkdir(os.path.join(root_dir, 'batch_3_partial'))
    for ba_t, ba_V in zip(batch_3_t, batch_3_V):
        ba_dir = os.path.join(os.path.join(root_dir, 'batch_3_partial'), ('batch_3_cell_' + str(index + 1)))
        os.mkdir(ba_dir)
        indexx = 0
        for t_cy, V_cy in zip(ba_t, ba_V):
            with open(os.path.join(ba_dir, ('batch_3_cell_' + str(index + 1) + '_cycle_' + str(indexx + 2)) + '.csv'),
                      'w', newline='') as f:
                writer = csv.writer(f)
                for t_, V_ in zip(t_cy, V_cy):
                    writer.writerow([t_, V_])
            indexx = indexx + 1
        index = index + 1

def main():
    batch_dir_root = r'G:\Battery dataset\batches'
    batch_1_cycle_dir = 'batch_1_cycle'
    batch_2_cycle_dir = 'batch_2_cycle'
    batch_3_cycle_dir = 'batch_3_cycle'
    path_1 = os.path.join(batch_dir_root, batch_1_cycle_dir)
    path_2 = os.path.join(batch_dir_root, batch_2_cycle_dir)
    path_3 = os.path.join(batch_dir_root, batch_3_cycle_dir)
    batch_1_list = os.listdir(path_1)
    batch_2_list = os.listdir(path_2)
    batch_3_list = os.listdir(path_3)
    batch_1_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_2_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))
    batch_3_list.sort(key=lambda l: int(re.findall('\d+', l)[1]))


    #battery_chose = os.path.join(path_1, batch_1_list[10])
    #partial_curve(battery_chose, False)

    writeflag = False
    t_all_partial, V_all_partial = [], []
    #, batch_2_list, batch_3_list


    #for path_i, i in zip([path_1, path_2, path_3], [batch_1_list, batch_2_list, batch_3_list]):
    for path_i, i in zip([path_1], [batch_1_list[0:2]]):
        t_batch, V_batch = [], []
        for j in i:
            t_ba, V_ba = [], []
            battery_chose = os.path.join(path_i, j)
            partial_curve(battery_chose, writeflag, t_ba, V_ba)
            t_batch.append(list(t_ba))
            V_batch.append(list(V_ba))
        t_all_partial.append(list(t_batch))
        V_all_partial.append(list(V_batch))
        #print(t_all_partial[0][0][0])

    if writeflag:
        csv_piece_write(t_all_partial, V_all_partial)


    plt.show()


if __name__ == '__main__':
    main()