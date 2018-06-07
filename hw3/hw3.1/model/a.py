import pickle,time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score

from util import generate, read_data, drop, get_center_gps, distance
import params

def fillna(train):
    print('fill na..')
    rncid = train.columns[train.columns.str.startswith('RNCID_')].tolist()
    cellid = train.columns[train.columns.str.startswith('CellID_')].tolist()
    asulevel = train.columns[train.columns.str.startswith('AsuLevel')].tolist()
    signallevel = train.columns[train.columns.str.startswith('SignalLevel')].tolist()
    rssi = train.columns[train.columns.str.startswith('RSSI_')].tolist()
    latitude  = train.columns[train.columns.str.startswith('Latitude_')].tolist()
    longitude  = train.columns[train.columns.str.startswith('Longitude_')].tolist()
    latitude, longitude = latitude[1:], longitude[1:]
    for i, row in train.iterrows():
        for (la, lo, rn, cell, asu, signal, rs) in zip(latitude, longitude, rncid, cellid, asulevel, signallevel, rssi):
            if np.isnan(row[la]) or np.isnan(row[lo]):
                # print(i, la, lo, rn, cell, asu, signal, rs)
                train.loc[i, la] = row[latitude[0]]
                train.loc[i, lo] = row[latitude[0]]
                train.loc[i, rn] = row[rncid[0]]
                train.loc[i, cell] = row[cellid[0]]
                train.loc[i, asu] = row[asulevel[0]]
                train.loc[i, signal] = row[signallevel[0]]
                train.loc[i, rs] = row[rssi[0]]
                # r = (r+1) % l
    return train


def load_train_test_split(train, n=10):
    splits = []
    # for seed in params.TRAIN_TEST_SPLITS[:n]:
    for seed in range(n):
        X_train, X_test, y_train, y_test = train_test_split(train.loc[:, ~train.columns.isin(['Latitude_m', 'Longitude_m', 'Target'])], train[['Latitude_m', 'Longitude_m', 'Target']],
                                                            test_size=0.2, random_state=seed)
        splits.append((X_train, X_test, y_train, y_test))
    return splits


def cell_evalution(label, pred):
    t = label.value_counts().to_dict()
    f = pd.Series(pred).value_counts()[:10].to_dict()
    cells = list(f.keys())
    d = {cell: 0 for cell in cells}
    df = pd.DataFrame({'target': label, 'pred': pred})
    for cell in cells:
        cell_df = df[df['target'] == cell]
        try:
            intersection = (cell_df['target'] == cell_df['pred']).value_counts()[True]
        except:
            intersection = 0
        d[cell] = intersection
    r = sum(d.values()) / sum([t.get(x, 0) for x in d])
    p = sum(d.values()) / sum([f.get(x, 0) for x in d])
    f = 2*p*r/(r+p) if r+p else 0
    # print('intersection:', sum(d.values()), 'n:', len(pred))
    return r, f, p, precision_score(label, pred, average='micro')


def distance_evaluation(test_pred, y_test):
    pred_gps = list(map(get_center_gps, test_pred))
    split_dist = []
    for i in range(len(pred_gps)):
        dis = distance(pred_gps[i][0], pred_gps[i][1], y_test.iloc[i, 0], y_test.iloc[i, 1])
        split_dist.append(dis)
    return split_dist


def get_average_dist(splits_dist):
    dist = []
    splits_dist = np.array(splits_dist)
    for idx in range(splits_dist.shape[1]):
        dist.append(np.mean(splits_dist[:, idx]))
    return dist


def plot():
    model_dist = pickle.load(open(params.errors_pkl, 'rb'))
    statics = pd.read_csv('../data/output/1452983_1i_statics.txt')

    fig = plt.figure(7, figsize=(30, 20))
    idx = 1
    fig.add_subplot(4, 2, idx)
    idx += 1

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
    for i in range(len(model_dist)):  #
        plt.scatter(np.where(model_dist == model_dist[i][len(model_dist[i])//2])[-1][0], model_dist[i][len(model_dist[i])//2], marker='*')
        plt.plot(range(len(model_dist[i])), model_dist[i], label=statics.loc[i, 'Name'], color=colors[i])
    plt.xlabel('No.')
    plt.ylabel('Error(meters)')
    plt.title('Distance Error Distribution')
    plt.legend()

    for col in ['Time', 'Recall', 'Precision', 'F', 'Overall Precision', 'Median Error']:
        fig.add_subplot(4, 2, idx)
        idx += 1
        plt.ylabel(col)
        plt.title(col)
        plt.grid()
        s = statics.sort_values(by=col)
        plt.scatter(s['Name'], s[col])
        plt.plot(s['Name'], s[col])
    plt.show()

def train(train, names, models):
    print('train:', train.shape, train.columns.tolist())

    model_cell_eval, model_dist, model_elapsed = [], [], []
    for name, model in zip(names, models):
        print(name, '=' * 30)
        splits_cell_eval, splits_time, splits_dist = [], [], []
        for i, (X_train, X_test, y_train, y_test) in enumerate(load_train_test_split(train, n=10)):
            start_time = time.time()
            model.fit(X_train, y_train['Target'])
            test_pred = model.predict(X_test)

            split_cell_val = cell_evalution(y_test['Target'], test_pred)
            split_dist = distance_evaluation(test_pred, y_test)

            elapsed_time = time.time() - start_time
            splits_cell_eval.append(split_cell_val)
            splits_dist.append(split_dist)
            splits_time.append(elapsed_time)
            print('S:{}  R:{:.3f} F:{:.3f} P:{:.3f} OP:{:.3f} T:{:.0f} s'.format(i + 1, *split_cell_val, elapsed_time))

        model_cell_eval.append(np.mean(np.array(splits_cell_eval), axis=0))
        model_elapsed.append(np.mean(splits_time))
        dist = get_average_dist(splits_dist)
        dist.sort()
        model_dist.append(dist)

        print('Middle error:{:.3f}  Min:{:.3f} Max:{:.3f}'.format(np.median(dist), min(dist), max(dist)))
        print('{}  R: {:.3f} F:{:.3f} P:{:.3f} OP:{:.3f} T:{:.0f} s'.format(name, *model_cell_eval[-1], model_elapsed[-1]))

    model_cell_eval = np.array(model_cell_eval)
    statics = pd.DataFrame(
        {'Name': names, 'Time': model_elapsed, 'Recall': model_cell_eval[:, 0], 'F': model_cell_eval[:, 1],
         'Precision': model_cell_eval[:, 2], 'Overall Precision': model_cell_eval[:, 3], 'Median Error': list(map(lambda x:x[len(x)//2], model_dist))})
    statics.sort_values(by=['Median Error'], ascending=False, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '1i', 'statics']) + '.txt', index=False)
    pickle.dump(model_dist, open(params.errors_pkl, 'wb'))
    return model_dist, statics


if __name__ == '__main__':
    train_flag = 1
    if train_flag:
        data = generate(fillna_with_0=True)
        # data = read_data()

        nb = GaussianNB()
        knn = KNeighborsClassifier(n_neighbors=3)
        dtc = DecisionTreeClassifier(random_state=0)

        ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
        rf = RandomForestClassifier()
        bagging = BaggingClassifier()
        dt = DecisionTreeClassifier()
        gbdt = GradientBoostingClassifier(n_estimators=10)

        names = ['GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'RandomForestClassifier', 'BaggingClassifier', 'GradientBoostingClassifier']
        models = [nb, knn, dt, ada, rf, bagging, gbdt]

        # data = drop(data, cols=['AsuLevel', 'SignalLevel', 'RNCID', 'CellID', 'IMSI', 'MRTime', '3', '4', '5', '6', '7'])
        data = drop(data, cols=['AsuLevel', 'SignalLevel', 'RNCID', 'CellID', 'IMSI', 'MRTime'])
        model_dist, statics = train(data, names, models)
    else:
        plot()


