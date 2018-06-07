import pickle, time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from math import sqrt,sin,cos,radians,asin
import params
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

gongcan = pd.read_csv(params.gongcan)
data_2g = pd.read_csv(params.data_2g)


def distance(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = radians(lng1),radians(lat1),radians(lng2),radians(lat2)
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis


def get_cell_label(data_2g, cell_len=20):
    print('divide cells..')
    lo, la = 'Longitude', 'Latitude'
    lo_min, lo_max, la_min, la_max = data_2g[lo].min(), data_2g[lo].max(), data_2g[la].min(), data_2g[la].max()
    print('lo_min {}, lo_max {}, la_min {}, la_max {}'.format(lo_min, lo_max, la_min, la_max))
    la_diff, lo_diff = distance(la_min, lo_min, la_max, lo_min), distance(la_min, lo_min, la_min, lo_max)
    rows, cols = math.ceil(la_diff / cell_len), math.ceil(lo_diff / cell_len)

    grid = np.arange(0, rows * cols).reshape(rows, cols)
    print('row {} col {} count {}'.format(rows, cols, rows * cols))

    data_labels = []
    for (lon, lat) in zip(data_2g[lo], data_2g[la]):
        lo_idx = math.floor(distance(la_min, lo_min, la_min, lon) / 20)
        la_idx = math.floor(distance(la_min, lo_min, lat, lo_min) / 20)
        data_labels.append(grid[la_idx][lo_idx])
    data_2g['Target'] = data_labels

    print(data_2g.columns)
    return data_2g


def get_cell_id(lat, lon):
    cell_len = 20
    lo, la = 'Longitude', 'Latitude'

    lo_min, lo_max, la_min, la_max = data_2g[lo].min(), data_2g[lo].max(), data_2g[la].min(), data_2g[la].max()
    la_diff, lo_diff = distance(la_min, lo_min, la_max, lo_min), distance(la_min, lo_min, la_min, lo_max)
    rows, cols = math.ceil(la_diff / cell_len), math.ceil(lo_diff / cell_len)

    grid = np.arange(0, rows * cols).reshape(rows, cols)

    lo_idx = math.floor(distance(la_min, lo_min, la_min, lon) / 20)
    la_idx = math.floor(distance(la_min, lo_min, lat, lo_min) / 20)
    return grid[la_idx][lo_idx]

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


def get_bs_gps(train):
    recid = data_2g.columns[data_2g.columns.str.startswith('RNCID')].tolist()
    cellid = data_2g.columns[data_2g.columns.str.startswith('CellID')].tolist()
    for i, left_on in enumerate(zip(recid, cellid)):
        print(i, train.shape)
        train = train.merge(gongcan[['RNCID', 'CellID', 'Longitude', 'Latitude']], how='left', left_on=left_on,
                            right_on=['RNCID', 'CellID'])
        train.drop(['RNCID', 'CellID'], inplace=True, axis=1)
        train = train.rename(columns={'Longitude': 'Longitude_' + str(i + 1), 'Latitude': 'Latitude_' + str(i + 1)})
    return train


def generate(fillna_with_0=True):
    train, helper = data_2g.copy(), gongcan.copy()
    train = get_cell_label(train)
    train = train.rename(columns={'Longitude': 'Longitude_m', 'Latitude': 'Latitude_m'})
    train = get_bs_gps(train)
    print(train.shape, train.columns.tolist())
    if fillna_with_0:
        train.fillna(0, inplace=True)
    else:
        train = fillna(train)
    pickle.dump(train, open(params.train_pkl, 'wb'))
    print('generate finished')
    return train


def read_data():
    try:
        train = pickle.load(open(params.train_file, 'rb'))
    except:
        print("data not found, creating them..")
        train = generate()
    finally:
        return train

#横纬la竖经lo
# row 65 col 82 count5329
def get_center_gps(idx):
    row = 65
    col = 82
    lo_min, lo_max, la_min, la_max = 121.20120490000001,121.2183295,31.28175691, 31.29339344
    la = (idx/col +0.5) / row * (la_max-la_min) + la_min
    lo = (idx%col + 0.5) / col * (lo_max-lo_min) + lo_min

    return [la, lo]



def load_train_test_split(train, n=10):
    splits = []
    # for seed in params.TRAIN_TEST_SPLITS[:n]:
    for seed in range(n):
        X_train, X_test, y_train, y_test = train_test_split(train.loc[:, ~train.columns.isin(['Latitude_m', 'Longitude_m', 'Target'])], train[['Latitude_m', 'Longitude_m', 'Target']],
                                                            test_size=0.2, random_state=seed)
        splits.append((X_train, X_test, y_train, y_test))
    return splits


def drop(data_2g, cols=['asulevel', 'signallevel', 'recid', 'cellid', 'IMSI', 'MRTime', 'Num_connected']):
    for each in cols:
        data_2g = data_2g.drop(data_2g.columns[data_2g.columns.str.find(each) != -1].tolist(), axis=1)
    return data_2g


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
    print('intersection:', sum(d.values()), 'n:', len(pred))
    return r, 2*p*r/(r+p), p, precision_score(label, pred, average='micro')



if __name__ == '__main__':
    train = generate(fillna_with_0=True)
    # train = read_data()
