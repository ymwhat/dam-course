import util, params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math, pickle
from c import load_train_test_split, get_relative_pos, distance_evaluation, get_median_errors, get_average_dist, plot_median_errors, plot

main_bs_cols = ['RNCID_1', 'CellID_1']
relaive_gps_cols=['relative_longitude', 'relative_latitude']

def get_k_data():
    dist_df = pickle.load(open(params.c_errors_pkl, 'rb'))
    median_errors = get_median_errors(dist_df)
    k = math.ceil(len(median_errors) * 0.2)
    top = np.array(sorted(median_errors.items(), key=lambda d: d[1], reverse=False)[:k])
    dap = np.array(sorted(median_errors.items(), key=lambda d: d[1], reverse=True)[:k])
    return top, dap

def get_K():
    dist_df = pickle.load(open(params.c_errors_pkl, 'rb'))
    median_errors = get_median_errors(dist_df)
    k = math.ceil(len(median_errors) * 0.2)
    return k

def train(data):
    splits_dist=[]
    top, dap = get_k_data()

    for i, (X_train, X_test, y_train, y_test) in enumerate(load_train_test_split(data, n=2)):
        print('Split:', i + 1)
        top_mr = []
        clfs = {}
        # get top-k data
        for name, group in X_train.groupby(main_bs_cols):
            if str(name) in top[:, 0]:
                top_mr = np.append(top_mr, group.index)
                print('top group:', len(group.index))
        top_data = X_train.loc[top_mr]

        #train
        for name, group in X_train.groupby(main_bs_cols):
            if str(name) in dap[:, 0]:
                print('dap group:', group.shape[0])
                group = pd.concat([group, top_data.iloc[:top_data.shape[0]]])
            group = util.drop(group, cols=['RNCID', 'CellID'])
            rf = RandomForestClassifier()
            rf.fit(group, y_train.loc[group.index][relaive_gps_cols].astype('int'))
            clfs[str(name)] = rf

        #predict
        test_pred = pd.DataFrame(index=X_test.index, columns=relaive_gps_cols)
        for name, group in X_test.groupby(main_bs_cols):
            group = util.drop(group, cols=['RNCID', 'CellID'])
            group_test_pred = clfs[str(name)].predict(group)
            test_pred.loc[group.index] = group_test_pred

        test_pred['pred_longitude'] = test_pred['relative_longitude']/pow(10, 20) + X_test['Longitude_1']
        test_pred['pred_latitude'] = test_pred['relative_latitude']/pow(10, 20) + X_test['Latitude_1']

        split_dist = distance_evaluation(test_pred, y_test)
        splits_dist.append(split_dist)

    dist_df = X_test[main_bs_cols]
    dist = get_average_dist(splits_dist)
    dist_df['dist'] = dist
    pickle.dump(dist_df, open(params.d_errors_pkl, 'wb'))


def plot_median_errors_comparison(file=[params.c_errors_pkl, params.d_errors_pkl]):
    plt.figure(figsize=(5, 5))
    colors = ['r', 'g', 'b', 'm', 'y']
    for f, color in zip(file, colors):
        # print(f)
        median_errors = get_median_errors(pickle.load(open(f, 'rb')))
        median_errors = np.array(sorted(median_errors.items(), key=lambda x: x[1]))
        plt.plot(range(median_errors.shape[0]), median_errors[:, 1].astype(float), linewidth=0.5, color=color, label=f, marker='o',
                 markerfacecolor='blue', markersize=1)

    plt.plot([median_errors.shape[0]- get_K()] * 100, range(0, 100))
    plt.grid()
    plt.xlabel('No.')
    plt.ylabel('Error(meters)')
    plt.legend()
    plt.title('Median Distance Error Distribution of Each group')
    plt.show()

    # ax = plt.gca()
    # for i in range(0, median_errors.shape[0], 10):
    #     x = i
    #     y = median_errors[:, 1].astype(float)[i]
    #     ax.annotate(round(y, 2), (x, y), xytext = (-4, 50))
    # for i in range(0, b_median_errors.shape[0], 10):
    #     x = i
    #     y = b_median_errors[:, 1].astype(float)[i]
    #     ax.annotate(round(y, 2), (x, y))



if __name__ == '__main__':
    train_flag = 0
    if train_flag:
        data = util.generate(fillna_with_0=True)
        data = get_relative_pos(data)
        data = util.drop(data, cols=['SignalLevel', 'AsuLevel',  'IMSI', 'MRTime'])
        train(data)
    else:
        plot_median_errors_comparison()
        # plot(params.d_errors_pkl)
        # plot(params.c_errors_pkl)
        # plot_median_errors(params.d_errors_pkl)
        # plot_median_errors(params.c_errors_pkl)









