import util, params
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math, pickle
from c import load_train_test_split, get_relative_pos, distance_evaluation, get_median_errors, get_average_dist, plot_median_errors, plot
from d import plot_median_errors_comparison
from sklearn.cluster import KMeans

main_bs_cols = ['RNCID_1', 'CellID_1']
relaive_gps_cols=['relative_longitude', 'relative_latitude']


def get_gps(data, bs):
    bs_gps = np.unique(data[main_bs_cols + ['Latitude_1', 'Longitude_1']], axis=0)
    idx = np.where(bs_gps[:,0] == bs[0]) and np.where(bs_gps[:,1] == bs[1])[0][0]
    # print(bs, bs_gps[idx])
    return bs_gps[idx][2:]


def get_top_data(dist_df, data):
    median_errors = get_median_errors(dist_df)
    k = math.ceil(len(median_errors) * 0.2)
    print('k:', k)
    median_errors = sorted(median_errors.items(), key=lambda d: d[1], reverse=True)
    dap = median_errors[:k]

    clf = KMeans(n_clusters=k)
    bs = np.unique(data[main_bs_cols + ['Latitude_1', 'Longitude_1']], axis=0)
    cluster_label = clf.fit_predict(bs[:,2:])
    bs = np.c_[bs, cluster_label]
    bs = pd.DataFrame(bs, columns=['RNCID_1', 'CellID_1', 'Latitude_1', 'Longitude_1', 'label'])

    dap_mr = {}
    for each in dap:
        labels = list(set(bs.loc[(bs['RNCID_1'] == eval(each[0])[0]) & (bs['CellID_1'] == eval(each[0])[1])]['label']))[:2]
        dap_mr[each[0]] = bs.loc[bs['label'].isin(labels), main_bs_cols]
        print(each[0], labels, len(dap_mr[each[0]]))

    return dap, dap_mr



def train(data):
    splits_dist=[]
    dist_df = pickle.load(open(params.c_errors_pkl, 'rb'))
    for i, (X_train, X_test, y_train, y_test) in enumerate(load_train_test_split(data, n=2)):
        print('Split:', i + 1)
        clfs = {}
        #train
        dap, dap_mr = get_top_data(dist_df, X_train)
        dap = np.array(dap)[:,0]
        for name, group in X_train.groupby(main_bs_cols):
            if str(name) in dap:
                print('before concat', group.shape)
                for n, g in X_train.groupby(['RNCID_1', 'CellID_1']):
                    if dap_mr[str(name)][(n[0] == dap_mr[str(name)]['RNCID_1']) & (n[1] == dap_mr[str(name)]['CellID_1'])].shape[0] :
                        group = pd.concat([group, g])
                print('after concat', group.shape)
            group = util.drop(group, cols=['RNCID', 'CellID'])
            rf = RandomForestClassifier()
            rf.fit(group, y_train.loc[group.index][relaive_gps_cols].astype('int'))
            clfs[str(name)] = rf

        test_pred = pd.DataFrame(index=X_test.index, columns=relaive_gps_cols)
        for name, group in X_test.groupby(main_bs_cols):
            group = util.drop(group, cols=['RNCID', 'CellID'])
            group_test_pred = clfs[str(name)].predict(group)
            test_pred.loc[group.index] = group_test_pred

        test_pred['pred_longitude'] = test_pred['relative_longitude']/pow(10, 20) + X_test['Longitude_1']
        test_pred['pred_latitude'] = test_pred['relative_latitude']/pow(10, 20) + X_test['Latitude_1']

        split_dist = util.distance_evaluation(test_pred, y_test)
        splits_dist.append(split_dist)

    dist_df = X_test[main_bs_cols]
    dist = get_average_dist(splits_dist)
    dist_df['dist'] = dist
    pickle.dump(dist_df, open(params.e_errors_pkl, 'wb'))


if __name__ == '__main__':
    train_flag = 0
    if train_flag:
        data = util.generate(fillna_with_0=True)
        data = get_relative_pos(data)
        print(data[relaive_gps_cols].isnull())
        data = util.drop(data, cols=['SignalLevel', 'AsuLevel',  'IMSI', 'MRTime'])
        train(data)
        plot(file=[params.c_errors_pkl, params.e_errors_pkl])
        plot_median_errors_comparison(file=[params.c_errors_pkl, params.d_errors_pkl, params.e_errors_pkl])

    else:
        plot(file=[params.c_errors_pkl, params.d_errors_pkl, params.e_errors_pkl])
        plot_median_errors_comparison(file=[params.c_errors_pkl, params.d_errors_pkl, params.e_errors_pkl])
        # plot_median_errors_comparison(file=[, params.e_errors_pkl])
        # plot(params.c_errors_pkl)
        # plot_median_errors(params.d_errors_pkl)
        # plot_median_errors(params.c_errors_pkl)
        # plot_median_errors(params.e_errors_pkl)




