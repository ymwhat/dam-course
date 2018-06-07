import util, params
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math, pickle

main_bs_cols = ['RNCID_1', 'CellID_1']
relaive_gps_cols=['relative_longitude', 'relative_latitude']

def get_relative_pos(data):
    data[relaive_gps_cols[0]] = (data['Longitude_m'] - data['Longitude_1'])*pow(10, 20)
    data[relaive_gps_cols[1]] = (data['Latitude_m'] - data['Latitude_1'])*pow(10, 20)
    return data

def load_train_test_split(train, n=10):
    cols=['relative_longitude', 'relative_latitude', 'Target', 'Latitude_m', 'Longitude_m']
    splits = []
    for seed in range(n):
        X_train, X_test, y_train, y_test = train_test_split(train.loc[:,~train.columns.isin(cols)], train[cols],
                                                            test_size=0.2, random_state=seed)
        splits.append((X_train, X_test, y_train, y_test))
    return splits


def distance_evaluation(test_pred, y_test):
    split_dist = []
    for idx in range(len(test_pred)):
        dis = util.distance(test_pred.iloc[idx, 3], test_pred.iloc[idx, 2], y_test.iloc[idx, 3], y_test.iloc[idx, 4])
        split_dist.append(dis)
    return split_dist


def get_average_dist(splits_dist):
    dist = []
    splits_dist = np.array(splits_dist)
    for idx in range(splits_dist.shape[1]):
        dist.append(np.mean(splits_dist[:, idx]))
    return dist


def get_median_errors(dist_df):
    median_errors = {}
    for g, dists in dist_df.groupby(main_bs_cols):
        dists = sorted(dists['dist'])
        median_error = dists[len(dists) // 2]
        median_errors[str(g)] = median_error
    return median_errors


def plot_median_errors(file=[params.c_errors_pkl]):
    for f in file:
        print(f)
        plt.figure(figsize=(8, 8))
        dist_df = pickle.load(open(f, 'rb'))
        median_errors = get_median_errors(dist_df)
        median_errors = np.array(sorted(median_errors.items(), key=lambda x: x[1]))
        print(median_errors)
        plt.plot(range(median_errors.shape[0]), median_errors[:, 1].astype(float), linewidth=0.5, color='g', marker='o',
                 markerfacecolor='blue', markersize=1)
        plt.grid()
        plt.xlabel('No.')
        plt.ylabel('Error(meters)')
        plt.title('Median Distance Error Distribution of Each group')

        ax = plt.gca()
        for i in range(0, median_errors.shape[0], 10):
            x = i
            y = median_errors[:, 1].astype(float)[i]
            ax.annotate(round(y, 2), (x, y))

    plt.show()


def plot(file=[params.c_errors_pkl]):
    plt.figure(figsize=(30, 8))
    colors = ['r', 'g', 'b', 'm', 'y']
    for f, color in zip(file, colors):
        # print(f)
        dist_df = pickle.load(open(f, 'rb'))
        median_errors = get_median_errors(dist_df)
        for g, dists in dist_df.groupby(main_bs_cols):
            dists = sorted(dists['dist'])
            plt.plot(range(len(dists)), dists, linewidth=0.3,  color=color,  marker='.', markersize=1)
            plt.scatter(len(dists) // 2, median_errors[str(g)], color='black',  marker='+')


    plt.xlabel('No.')
    plt.ylabel('Error(meters)')
    plt.title('Distance Error Distribution')
    plt.grid()
    plt.show()

def train(data):
    splits_dist = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(load_train_test_split(data, n=3)):
        clfs = {}
        for name, group in X_train.groupby(main_bs_cols):
            group = util.drop(group, cols=['RNCID', 'CellID'])
            rf = RandomForestClassifier()
            rf.fit(group, y_train.loc[group.index][relaive_gps_cols].astype('int'))
            clfs[str(name)] = rf

        test_pred = pd.DataFrame(index=X_test.index, columns=relaive_gps_cols)
        for name, group in X_test.groupby(main_bs_cols):
            group = util.drop(group, cols=['RNCID', 'CellID'])
            group_test_pred = clfs[str(name)].predict(group)
            test_pred.loc[group.index] = group_test_pred

        test_pred['pred_longitude'] = test_pred[relaive_gps_cols[0]] / pow(10, 20) + X_test['Longitude_1']
        test_pred['pred_latitude'] = test_pred[relaive_gps_cols[1]] / pow(10, 20) + X_test['Latitude_1']

        split_dist = distance_evaluation(test_pred, y_test)
        splits_dist.append(split_dist)
    dist_df = X_test[main_bs_cols]
    dist = get_average_dist(splits_dist)
    dist_df['dist'] = dist
    pickle.dump(dist_df, open(params.c_errors_pkl, 'wb'))



if __name__ == '__main__':
    train_flag = 0
    if train_flag:
        data = util.generate(fillna_with_0=True)
        data = get_relative_pos(data)
        data = util.drop(data, cols=['SignalLevel', 'AsuLevel', 'IMSI', 'MRTime'])
        train(data)
    else:
        plot()
        plot_median_errors()













