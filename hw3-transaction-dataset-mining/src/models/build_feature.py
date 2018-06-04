import params
import numpy as np
import pandas as pd
import ast
from functools import reduce
from util import pickle_load, pickle_dump

class Feature(object):
    def __init__(self):
        print('Feature init..')
        self.data = self.read_raw_data()

    def __init__(self, read_file_name='../../data/raw/trade_new.csv', month=[2, 3, 4], train=True, save_pos='/234'):
        self.read_file_name = read_file_name
        self.month = month
        data = self.read_raw_data(read_file_name)
        self.data = data
        if train:
            path = params.pkl_train_path + save_pos
        else:
            path = params.pkl_test_path + save_pos
        try:
            print('Feature loading ... from ', path, ' month: ', month)
            self.user_bci_agg = pickle_load(path+'/user_bci_agg.pkl')
            self.bci_user_agg = pickle_load(path+'/bci_user_agg.pkl')

            self.m_action_cr_agg = pickle_load(path+'/m_action_cr_agg.pkl')
            self.m_action_cr = pickle_load(path+'/m_action_cr.pkl')

            self.m_pen_cr_agg = pickle_load(path+'/m_pen_cr_agg.pkl')
            self.m_pen_cr = pickle_load(path+'/m_pen_cr.pkl')

            self.m_pd_cr_agg = pickle_load(path+'/m_pd_cr_agg.pkl')
            self.m_pd_cr = pickle_load(path+'/m_pd_cr.pkl')

            self.repeat = pickle_load(path+'/repeat.pkl')

            self.items = pickle_load( path+'/items.pkl')
            self.users = pickle_load(path+'/users.pkl')
            self.items = pickle_load(path+'/items.pkl')
            self.brands = pickle_load(path+'/brands.pkl')
            self.cats = pickle_load(path+'/cats.pkl')
            print('loading finished')

        except:
            print('init..')
            user_bci_agg = bci_agg(data, months=month, groupby1=['user_id'], groupby2=['item_id'])
            bci_user_agg = user_agg(data, months=month, groupby=['brand_id', 'cat_id', 'item_id'])
            self.user_bci_agg = user_bci_agg
            self.bci_user_agg = bci_user_agg

            pickle_dump(self.user_bci_agg, path+'/user_bci_agg.pkl')
            pickle_dump(self.bci_user_agg, path+'/bci_user_agg.pkl')

            m_action_cr = monthly_action_cr(data, month)
            m_action_cr_agg = monthly_action_cr_agg(m_action_cr)
            self.m_action_cr_agg = m_action_cr_agg
            self.m_action_cr = m_action_cr

            pickle_dump(self.m_action_cr_agg, path+'/m_action_cr_agg.pkl')
            pickle_dump(self.m_action_cr, path+'/m_action_cr.pkl')


            m_pen_cr = monthly_penetration_cr(data, month, groupby=['brand_id', 'cat_id', 'item_id'])
            m_pen_cr_agg = penetration_agg(m_pen_cr)
            self.m_pen_cr_agg = m_pen_cr_agg
            self.m_pen_cr = m_pen_cr

            pickle_dump(self.m_pen_cr_agg, path+'/m_pen_cr_agg.pkl')
            pickle_dump(self.m_pen_cr, path+'/m_pen_cr.pkl')

            m_pd_cr = monthly_product_diversity_cr(data, months=month, groupby=['user_id', 'brand_id', 'cat_id'])
            k_attrs = {
                'user_id': ['item'],  # ['cat', 'brand', 'item'],
                'brand_id': ['cat', 'item'],
                'cat_id': ['brand', 'item']
            }
            m_pd_cr_agg = monthly_product_diversity_agg(m_pd_cr, k_attrs)
            self.m_pd_cr_agg = m_pd_cr_agg
            self.m_pd_cr = m_pd_cr

            pickle_dump(self.m_pd_cr_agg, path+'/m_pd_cr_agg.pkl')
            pickle_dump(self.m_pd_cr, path+'/m_pd_cr.pkl')

            self.repeat = repeat_feature(data, months=[2, 3, 4], groupby=['brand_id', 'cat_id', 'item_id', 'user_id'])
            pickle_dump(self.repeat, path+'/repeat.pkl')

            self.item_profile()
            self.user_profile()
            self.brand_profile()
            self.cat_profile()

            pickle_dump(self.items, path+'/items.pkl')
            pickle_dump(self.users, path+'/users.pkl')
            pickle_dump(self.items, path+'/items.pkl')
            pickle_dump(self.brands, path+'/brands.pkl')
            pickle_dump(self.cats, path+'/cats.pkl')

            print('init finished')
        finally:
            print('finished..')

    def read_raw_data(self, file_name):
        data = pd.read_csv(file_name, index_col=0)
        if data.columns.contains('sldatime'):
            data = data.rename(columns={'sldatime': 'sldat'})
        data['sldat'] = pd.to_datetime(data['sldat'], errors='coerce')
        data['bndno'] = data['bndno'].fillna(-1).astype('int', errors='ignore')
        data['pluno'] = data['pluno'].astype('int', errors='ignore')
        data['dptno'] = data['dptno'].astype('int', errors='ignore')
        data['month'] = data['sldat'].apply(lambda x: x.month)
        data['day'] = data['sldat'].apply(lambda x: x.day)
        data = data.rename(columns={'pluno': 'item_id', 'dptno': 'cat_id', 'bndno': 'brand_id', 'vipno': 'user_id'})
        print("read data finished")
        return data

    def item_profile(self):
        merge_key = 'item_id'
        trend = self.m_action_cr[merge_key].columns[
            self.m_action_cr[merge_key].columns.str.find('trend') != -1].tolist()
        dev = self.m_action_cr[merge_key].columns[self.m_action_cr[merge_key].columns.str.find('dev') != -1].tolist()
        repeat = self.repeat[merge_key].columns[self.repeat[merge_key].columns.str.find('repeat') != -1].tolist()

        items = self.m_action_cr_agg[merge_key].merge(self.m_pen_cr_agg[merge_key], on=merge_key, how='outer')\
            .merge(self.bci_user_agg[merge_key], on=merge_key, how='outer')\
            .merge(self.m_action_cr[merge_key][[merge_key]+trend+dev], on=merge_key, how='outer')\
            .merge(self.repeat[merge_key][[merge_key]+repeat], on=merge_key, how='outer')

        self.items = items
        del merge_key
        return self.items

    def user_profile(self):
        merge_key = 'user_id'
        trend = self.m_action_cr[merge_key].columns[self.m_action_cr[merge_key].columns.str.find('trend')!=-1].tolist()
        dev = self.m_action_cr[merge_key].columns[self.m_action_cr[merge_key].columns.str.find('dev')!=-1].tolist()

        users = self.m_action_cr_agg[merge_key].merge(self.m_pd_cr_agg[merge_key], on=merge_key, how='outer')\
            .merge(self.user_bci_agg[merge_key], on=merge_key, how='outer').merge(self.m_action_cr[merge_key][[merge_key]+trend+dev], on=merge_key, how='outer')
        self.users = users
        del merge_key

        return self.users

    def brand_profile(self):
        merge_key = 'brand_id'
        trend = self.m_action_cr[merge_key].columns[
            self.m_action_cr[merge_key].columns.str.find('trend') != -1].tolist()
        dev = self.m_action_cr[merge_key].columns[self.m_action_cr[merge_key].columns.str.find('dev') != -1].tolist()
        repeat = self.repeat[merge_key].columns[self.repeat[merge_key].columns.str.find('repeat') != -1].tolist()

        brands = self.m_action_cr_agg[merge_key].merge(self.m_pen_cr_agg[merge_key], on=merge_key, how='outer')\
            .merge(self.bci_user_agg[merge_key],on=merge_key, how='outer')\
            .merge(self.m_pd_cr_agg[merge_key], on=merge_key, how='outer') \
            .merge(self.m_action_cr[merge_key][[merge_key] + trend + dev], on=merge_key, how='outer') \
            .merge(self.repeat[merge_key][[merge_key] + repeat], on=merge_key, how='outer')

        self.brands = brands

        return self.brands

    def cat_profile(self):
        merge_key = 'cat_id'
        trend = self.m_action_cr[merge_key].columns[
            self.m_action_cr[merge_key].columns.str.find('trend') != -1].tolist()
        dev = self.m_action_cr[merge_key].columns[self.m_action_cr[merge_key].columns.str.find('dev') != -1].tolist()
        repeat = self.repeat[merge_key].columns[self.repeat[merge_key].columns.str.find('repeat') != -1].tolist()

        cats = self.m_action_cr_agg[merge_key].merge(self.m_pen_cr_agg[merge_key], on=merge_key, how='outer')\
            .merge(self.bci_user_agg[merge_key],on=merge_key, how='outer')\
            .merge(self.m_pd_cr_agg[merge_key], on=merge_key, how='outer') \
            .merge(self.m_action_cr[merge_key][[merge_key] + trend + dev], on=merge_key, how='outer') \
            .merge(self.repeat[merge_key][[merge_key] + repeat], on=merge_key, how='outer')

        self.cats = cats

        return self.cats

    def user_join_item(self):
        users = self.users
        items = self.items

        ui_df = self.m_action_cr_agg[str(['user_id', 'item_id'])].merge(users, on='user_id', how='left') \
            .merge(items, on='item_id', how='left')
        ui_df.fillna(0, inplace=True)
        return ui_df

    def user_join_brand(self):
        users = self.users
        brands = self.brands

        ub_df = self.m_action_cr_agg[str(['user_id', 'brand_id'])].merge(users, on='user_id', how='left') \
            .merge(brands, on='brand_id', how='left')
        ub_df.fillna(0, inplace=True)
        return ub_df

    def user_join_cat(self):
        users = self.users
        cats = self.cats

        ub_df = self.m_action_cr_agg[str(['user_id', 'cat_id'])].merge(users, on='user_id', how='left') \
            .merge(cats, on='cat_id', how='left')
        ub_df.fillna(0, inplace=True)
        return ub_df
    def user_item_label(self, df, month, cols):
        label_month = get_data_by_month(self.data, [month])
        label_month = label_month[cols].groupby(cols).count().reset_index()
        label_month['target'] = 1
        df = df.merge(label_month, on=cols, how='left')
        df.fillna(0, inplace=True)

        print(label_month.shape, df.shape, df['target'].value_counts())
        return df

    def user_item_data(self, file_name, label=True):
        cols = ['user_id', 'item_id']
        ui = self.user_join_item()
        if label:
            ui = self.user_item_label(ui, self.month[-1]+1, cols)

        ui = ui.merge(self.m_action_cr_agg[str(cols)], on=cols, how='left')
        ui.fillna(0, inplace=True)
        ui.to_csv(file_name, index=False)
        ui[['user_id', 'item_id', 'target']].to_csv(file_name + 'label', index=False)
        print('user_item_data:', ui.columns)
        return ui

    #repeat buyer
    def user_data(self, file_name):
        users = self.users #(416, 29)

        def get_m_days(m):
            cols = self.m_action_cr['user_id'].columns
            col = cols[cols.str.find(str(m)) != -1] & cols[cols.str.find(str('day')) != -1]
            col = col.tolist()
            return self.m_action_cr['user_id'][['user_id']+ col], col[0]

        def get_m_data(m):
            cols = self.m_action_cr['user_id'].columns
            col = cols[cols.str.find(str(m)) != -1].tolist()
            return self.m_action_cr['user_id'][['user_id']+ col]

        results = []
        if len(self.month) > 1:
            for m in self.month[:-1]:
                m_data = get_m_data(m)
                next_month, col_name = get_m_days(m+1)
                next_month = next_month.loc[next_month[col_name] != 0]
                next_month[col_name] = 1
                result = m_data.merge(next_month, on='user_id', how='left')
                result.fillna(0, inplace=True)
                result.columns=['user_id', 'frequency', 'days', 'amts', 'target']
                results.append(result)
            results = pd.concat(results)
            results = results.merge(users, on='user_id', how='left')
        else:
            results = get_m_data(self.month[0])
            if self.month[0] == 5:
                file = '/234'
            elif self.month[0] == 7:
                file='/456'
            else:
                print('error..')

            history_users = pickle_load(params.pkl_train_path + file + '/users.pkl')
            results['target'] = 1
            results = history_users.merge(results, on='user_id', how='left')
            results.fillna(0, inplace=True)
        results.to_csv(file_name, index=False)

        return users

    def user_amt_data(self, file_name):
        users = self.users #(416, 29)

        def get_m_amts(m):
            cols = self.m_action_cr['user_id'].columns
            col = cols[cols.str.find(str(m)) != -1] & cols[cols.str.find(str('amt')) != -1]
            col = col.tolist()
            return self.m_action_cr['user_id'][['user_id']+ col], col[0]

        def get_m_data(df, m):
            cols = df['user_id'].columns
            col = cols[cols.str.find(str(m)) != -1].tolist()
            return df['user_id'][['user_id']+ col]

        results = []
        if len(self.month) > 1:
            for m in self.month[:-1]:
                m_data = get_m_data(self.m_action_cr, m)
                next_month, col_name = get_m_amts(m+1)
                print(col_name)
                # next_month = next_month.loc[next_month[col_name] != 0]
                # next_month[col_name] = 1
                result = m_data.merge(next_month, on='user_id', how='left')
                result.fillna(0, inplace=True)
                result.columns=['user_id', 'frequency', 'days', 'amts', 'target']
                results.append(result)
                print(len(results))
            results = pd.concat(results)
            results = results.merge(users, on='user_id', how='left')
        else:
            cols = self.m_action_cr['user_id'].columns
            col = cols[cols.str.find(str(self.month[0])) != -1] & cols[cols.str.find(str('amt')) != -1]
            col = col.tolist()[0]
            results = get_m_data(self.m_action_cr, self.month[0])
            results = results.rename(columns={col:'target'})

            last_month = get_m_data(Feature(month=[self.month[0]-1], save_pos='/'+str(self.month[0]-1)).m_action_cr, self.month[0]-1)

            results = results[['user_id', 'target']].merge(last_month[last_month.columns.tolist()], on='user_id', how='left')
            results.columns = ['user_id', 'target', 'frequency', 'days', 'amts']

            if self.month[0] == 5:
                file = '/234'
            elif self.month[0] == 7:
                file = '/456'
            else:
                print('error..')

            history_users = pickle_load(params.pkl_train_path + file + '/users.pkl')
            results = history_users.merge(results, on='user_id', how='left')
            results.fillna(0, inplace=True)
            # print(results.shape)
        print('user data ', results.shape)
        results.to_csv(file_name, index=False)

        return users

    def user_brand_data(self, file_name, label=True):
        cols = ['user_id', 'brand_id']
        ub = self.user_join_brand()
        if label:
            ub = self.user_item_label(ub, self.month[-1] + 1, cols)

        #user_item record process
        ub = ub.merge(self.m_action_cr_agg[str(cols)], on=cols, how='left')
        ub.fillna(0, inplace=True)
        print('user brand data: ', ub.shape)
        ub.to_csv(file_name, index=False)
        return ub

    def user_cat_data(self, file_name, label=True):
        cols = ['user_id', 'cat_id']
        uc = self.user_join_cat()
        if label:
            uc = self.user_item_label(uc, self.month[-1] + 1, cols)

        # user_item record process
        # uc = uc.merge(self.m_action_cr_agg[str(cols)], on=cols, how='left')
        uc.fillna(0, inplace=True)
        print('user cat data: ', uc.shape)
        uc.to_csv(file_name, index=False)
        return uc

def get_data_by_month(data, month=[2, 3, 4]):
    if type(month) is int:
        t = data['sldat'].apply(lambda x: x.month)
        return data.loc[t.isin([month])]
    elif type(month) is list:
        t = data['sldat'].apply(lambda x: x.month)
        return data.loc[t.isin(month)]


def foo(x, cols):
    n = len(cols)
    alpha = (0.0 + n * np.sum([i * x[c] for i, c in enumerate(cols, 1)]) - np.sum(range(1, n + 1, 1)) * np.sum(
        x[cols])) / \
            (np.sum(np.square(range(1, n + 1, 1))) - np.square(np.sum(range(1, n + 1, 1))))
    deviation = [x[cols[-1]] - np.mean(x[cols]) / np.mean(x[cols]) if np.mean(x[cols]) else 0][0]
    return [alpha, deviation]


'''
monthly action count/ratio monthly_cr
'''
def monthly_action_cr(data,
                      months,
                      groupby=['user_id', 'brand_id', 'cat_id', 'item_id', ['user_id', 'brand_id'],
                               ['user_id', 'cat_id'], ['user_id', 'item_id']],
                      prefixes=['u_', 'b_', 'c_', 'i_', 'ub_', 'uc_', 'ui_'],
                      agg={'sldat': 'count', 'day': pd.Series.nunique, 'amt': 'sum'}
                      ):
    dfs_final = {str(k): [] for k in groupby}
    for g, prefix in zip(groupby, prefixes):
        print(g, prefix)
        dfs = []
        for month in months:
            name = prefix + 'count_' + str(month) + '_'
            m_g = get_data_by_month(data, month) \
                .groupby(g) \
                .agg(agg) \
                .reset_index() \
                .rename(columns={'sldat': name + 'frequency', 'day': name + 'days', 'amt': name + 'amts'})
            dfs.append(m_g)
        df_final = reduce(lambda left, right: left.merge(right, on=g, how='outer'), dfs)
        df_final.fillna(0, inplace=True)

        for col in ['frequency', 'days', 'amts']:
            print(g, prefix, col)
            name1, name2 = prefix + col + '_trend', prefix + col + '_dev'
            m_cols = df_final.columns[df_final.columns.str.find(col) != -1]
            # print(m_cols)
            alpha_dev = df_final[m_cols].apply(lambda x: foo(x, m_cols), axis=1)
            alpha = np.array(alpha_dev.tolist())[:, 0]
            dev = np.array(alpha_dev.tolist())[:, 1]
            df_final[name1] = alpha
            df_final[name2] = dev
            df_final.fillna(0, inplace=True)

        dfs_final[str(g)] = df_final
    return dfs_final


'''
count/ratio: monthly product diversity
groupby=['user_id', 'brand_id', 'cat_id']
'''
def monthly_product_diversity_cr(data: pd.DataFrame, months, groupby):
    dfs_final = {g: pd.DataFrame() for g in groupby}
    unq = pd.Series.nunique
    for g in groupby:
        dfs = []
        for month in months:
            prefix = g[0] + '_pd_' + str(month) + '_'
            if g == 'user_id':
                m_g = get_data_by_month(data, month).groupby(g) \
                    .agg({'cat_id': unq, 'brand_id': unq, 'item_id': unq}).reset_index() \
                    .rename(
                    columns={'cat_id': prefix + 'cat', 'brand_id': prefix + 'brand', 'item_id': prefix + 'item'})
            elif g == 'brand_id':
                m_g = get_data_by_month(data, month).groupby(g) \
                    .agg({'cat_id': unq, 'item_id': unq}).reset_index() \
                    .rename(columns={'cat_id': prefix + 'cat', 'item_id': prefix + 'item'})
            elif g == 'cat_id':
                m_g = get_data_by_month(data, month).groupby(g).agg({'brand_id': unq, 'item_id': unq}).reset_index(). \
                    rename(columns={'brand_id': prefix + 'brand', 'item_id': prefix + 'item'})
            dfs.append(m_g)

        df_final = reduce(lambda left, right: left.merge(right, on=g, how='outer'), dfs)
        df_final.fillna(0, inplace=True)

        if g == 'user_id':
            g_cols = ['cat', 'brand', 'item']
        elif g == 'brand_id':
            g_cols = ['cat', 'item']
        elif g == 'item_id':
            g_cols = ['brand', 'item']
        for col in g_cols:
            name1, name2 = g[0] + '_' + col + '_trend', g[0] + '_' + col + '_dev'

            m_cols = df_final.columns[df_final.columns.str.find(col) != -1]
            alpha_dev = df_final[m_cols].apply(lambda x: foo(x, m_cols), axis=1)
            alpha = np.array(alpha_dev.tolist())[:, 0]
            dev = np.array(alpha_dev.tolist())[:, 1]
            df_final[name1] = alpha
            df_final[name2] = dev
            df_final.fillna(0, inplace=True)

        dfs_final[g] = df_final
    return dfs_final


def repeat_feature(data: pd.DataFrame, months, groupby):
    dfs_final = {g: pd.DataFrame() for g in groupby}
    unq = pd.Series.nunique
    for g in groupby:
        if g == 'user_id':
            cols = ['item_id', 'brand_id', 'cat_id']
            g_data = get_data_by_month(data, months)
            g_df = pd.DataFrame({g: g_data[g].unique()})
            for col in cols:
                prefix = g[0] + col[0] + '_'
                repeat_day_col = g[0] + col[0] + '_repeat_day'

                name = prefix + 'repeat_count'
                # bci被每个User购买的unique day数目
                repeat_day_df = g_data.groupby([g, col]).agg({'day': pd.Series.nunique}).rename(
                    columns={'day': repeat_day_col})
                # bci被所有User购买的unique day之和
                repeat_day_denominator = int(repeat_day_df.agg({repeat_day_col: 'sum'}))
                repeat_bci_count_denominator = repeat_day_df.reset_index(level=1).groupby(level=0).agg(
                    {col: 'count'}).reset_index()
                # bci被重复购买的User购买的unique day数目
                repeat_day_df = repeat_day_df.loc[repeat_day_df[repeat_day_col] > 1]
                # repeat day of {I, B, C}的count
                repeat_day_count = repeat_day_df.reset_index(level=1).groupby(level=0).agg(
                    {repeat_day_col: 'sum'}).reset_index()

                repeat_bci_count = repeat_day_df.reset_index(level=1).groupby(level=0).agg({col: 'count'}).rename(
                    columns={col: name}).reset_index()

                g_df = g_df.merge(repeat_day_count, on=g, how='left').merge(repeat_bci_count, on=g, how='left')
                g_df.fillna(0, inplace=True)
                g_df[name + '_ratio'] = g_df.apply(lambda x: x[repeat_day_col] / int(
                    repeat_bci_count_denominator.loc[repeat_bci_count_denominator[g] == x[g], col]), axis=1)
                g_df[repeat_day_col + '_ratio'] = g_df.apply(lambda x: x[repeat_day_col] / repeat_day_denominator,
                                                             axis=1)

        else:
            name = g[0] + '_repeat_buyer'
            repeat_day_col = g[0] + '_repeat_day'
            g_data = get_data_by_month(data, months)
            g_df = pd.DataFrame({g: g_data[g].unique()})
            repeat_day_df = g_data.groupby([g, 'user_id']).agg({'day': pd.Series.nunique}).rename(
                columns={'day': repeat_day_col})
            repeat_day_denominator = repeat_day_df.groupby(level=0).agg({repeat_day_col: 'sum'})
            repeat_day_df = repeat_day_df.loc[repeat_day_df[repeat_day_col] > 1]

            repeat_buyer_count = repeat_day_df.reset_index(level=1).groupby(level=0).agg(
                {'user_id': pd.Series.nunique}).rename(columns={'user_id': name}).reset_index()
            repeat_buyer_denominator = g_data.groupby([g]).agg({'user_id': pd.Series.nunique})

            repeat_day_count = repeat_day_df.groupby(level=0).agg('sum').reset_index()
            g_df = g_df.merge(repeat_buyer_count, on=g, how='left').merge(repeat_day_count, on=g, how='left')
            g_df.fillna(0, inplace=True)
            g_df[name + '_ratio'] = g_df.apply(lambda x: x[name] / repeat_buyer_denominator.loc[x[g]], axis=1)
            g_df[repeat_day_col + '_ratio'] = g_df.apply(lambda x: x[name] / repeat_day_denominator.loc[x[g]], axis=1)
        dfs_final[g] = g_df
    return dfs_final


'''
count/ratio: monthly penetration
groupby=['user_id', 'brand_id', 'cat_id']
'''
def monthly_penetration_cr(data: pd.DataFrame, months, groupby):
    dfs_final = {k: [] for k in groupby}
    for g in groupby:
        dfs = []
        for month in months:
            prefix = g[0] + '_pen_' + str(month) + '_'
            m_g = get_data_by_month(data, month).groupby(g).agg({'user_id': pd.Series.nunique}).reset_index() \
                .rename(columns={'user_id': prefix + 'users'})
            dfs.append(m_g)
        df_final = reduce(lambda left, right: left.merge(right, on=g, how='outer'), dfs)

        col = 'user'
        name1, name2 = prefix + col + '_trend', prefix + col + '_dev'
        m_cols = df_final.columns[df_final.columns.str.find(col) != -1]
        alpha_dev = df_final[m_cols].apply(lambda x: foo(x, m_cols), axis=1)
        alpha = np.array(alpha_dev.tolist())[:, 0]
        dev = np.array(alpha_dev.tolist())[:, 1]
        df_final[name1] = alpha
        df_final[name2] = dev
        df_final.fillna(0, inplace=True)
        dfs_final[g] = df_final

    return dfs_final


'''
aggregation : product diversity
k_attrs = {
    'user_id': ['item'],  # ['cat', 'brand', 'item'],
    'brand_id': ['cat', 'item'],
    'cat_id': ['brand', 'item']
}
'''
def monthly_product_diversity_agg(d: dict, k_attrs: dict, agg_option=['sum', 'min', 'max', 'mean']):
    dfs_final = {k: pd.DataFrame() for k in d}
    for k in d:
        attrs = {i: d[k].columns[d[k].columns.str.find(i) != -1].tolist() for i in k_attrs[k]}
        dfs = []
        for attr in attrs:
            prefix = k[0] + '_monthly_pd_' + attr
            agg = d[k][attrs[attr] + [k]].set_index([k]).stack().groupby(level=0) \
                .agg(agg_option) \
                .rename(columns={'sum': prefix + '_sum', 'min': prefix + '_min', 'max': prefix + '_max',
                                 'mean': prefix + '_mean'})
            dfs.append(agg)
        df_final = reduce(lambda left, right: left.join(right), dfs).reset_index()

        dfs_final[k] = df_final
    return dfs_final


'''
aggregation : penetration
'''
def penetration_agg(d: dict, agg_option=['sum', 'min', 'max', 'mean']):
    dfs_final = {k: pd.DataFrame() for k in d}

    for k in d:
        dfs = []
        attrs = {i: d[k].columns[d[k].columns.str.find('users') != -1].tolist() for i in ['users']}
        for attr in attrs:
            name_prefix = k[0] + '_monthly_pen_' + attr
            agg = d[k][attrs[attr] + [k]].set_index([k]).stack().groupby(level=0).agg(agg_option) \
                .rename(columns={'sum': name_prefix + '_sum', 'min': name_prefix + '_min', 'max': name_prefix + '_max',
                                 'mean': name_prefix + '_mean'})
            dfs.append(agg)
        df_final = reduce(lambda left, right: left.join(right, lsuffix='_l', rsuffix='r_'), dfs).reset_index()
        df_final
        dfs_final[k] = df_final
    return dfs_final


'''
aggregation : monthly aggregation
'''
def monthly_action_cr_agg(d, agg_options=['sum', 'min', 'max', 'mean']):
    prefixes = ['u_', 'b_', 'c_', 'i_', 'ub_', 'uc_', 'ui_']
    groupby = ['user_id', 'brand_id', 'cat_id', 'item_id', ['user_id', 'brand_id'], ['user_id', 'cat_id'],
               ['user_id', 'item_id']]

    dfs_final = {k: pd.DataFrame() for k in d}
    for k, pf in zip(groupby, prefixes):
        k = str(k)
        attrs = {i: d[k].columns[d[k].columns.str.find(i) != -1].tolist() for i in ['frequency', 'days', 'amts']}
        dfs = []
        for attr in attrs:
            print(k, pf, attr)
            prefix = pf + 'monthly_count_' + attr
            if k.find(',') == -1:
                agg = d[k][attrs[attr] + [k]].set_index(k) \
                    .stack().groupby(level=0).agg(agg_options) \
                    .rename(columns={'sum': prefix + '_sum', 'min': prefix + '_min', 'max': prefix + '_max',
                                     'mean': prefix + '_mean'})
            else:
                list_k = ast.literal_eval(k)
                agg = d[k][attrs[attr] + list_k] \
                    .set_index(list_k).stack().groupby(level=[0, 1]) \
                    .agg(agg_options) \
                    .rename(columns={'sum': prefix + '_sum', 'min': prefix + '_min', 'max': prefix + '_max',
                                     'mean': prefix + '_mean'})
            dfs.append(agg)
        df_final = reduce(lambda left, right: left.join(right), dfs).reset_index()
        dfs_final[str(k)] = df_final
    return dfs_final


'''
aggregation : user aggregation
'''
def user_agg(d: pd.DataFrame, months=[2, 3, 4], groupby=['brand_id', 'cat_id', 'item_id'],
             agg=['sum', 'min', 'max', 'mean']):
    dfs_final = {k: pd.DataFrame() for k in groupby}
    data = get_data_by_month(d, months)
    for g1 in groupby:
        g2 = 'user_id'
        prefix = ''
        t = data.groupby([g1, g2]).agg({'day': pd.Series.nunique, 'sldat': 'count', 'amt': 'sum'}).reset_index()
        t = t.groupby(g1).agg({'day': agg, 'sldat': agg, 'amt': agg}) \
            .rename(columns={'sum': prefix + 'sum', 'min': prefix + 'min', 'max': prefix + 'max',
                             'mean': prefix + 'mean'}).reset_index()
        col_name = ["_".join(x) for x in t.columns.ravel()]
        col_name[0] = col_name[0][:-1]
        t.columns = col_name

        dfs_final[g1] = t
    return dfs_final

'''
aggregation : brand/category/item aggregation
'''
def bci_agg(d: pd.DataFrame, months=[2, 3, 4], groupby1=['user_id'], groupby2=['brand_id', 'cat_id', 'item_id'],
            agg=['sum', 'min', 'max', 'mean']):
    dfs_final = {k: pd.DataFrame() for k in groupby1}
    data = get_data_by_month(d, months)
    for g1 in groupby1:
        for g2 in groupby2:
            prefix = ''
            print(prefix)
            t = data.groupby([g1, g2]).agg({'day': pd.Series.nunique, 'sldat': 'count', 'amt': 'sum'}).reset_index()

            t = t.groupby(g1).agg({'day': agg, 'sldat': agg, 'amt': agg}) \
                .rename(columns={'sum': prefix + 'sum', 'min': prefix + 'min', 'max': prefix + 'max',
                                 'mean': prefix + 'mean'})
            col_name = ["_".join(x) for x in t.columns.ravel()]
            col_name = [g2[0] + '_' + x for x in col_name]
            t.columns = col_name
            t = t.reset_index()
            t = t.rename(columns={'user_id/': 'user_id'})
            dfs_final[g1] = t
    return dfs_final

def train_csv(month=[4, 5, 6], save_pos='/456'):
    train = Feature(month=month, save_pos=save_pos)
    train_path = params.train_path + save_pos
    print(train_path)
    # train.user_item_data(train_path + params.b_train_file_name)
    # train.user_data(train_path + params.ci_train_file_name)
    # train.user_amt_data(train_path + params.civ_train_file_name)
    # train.user_brand_data(train_path + params.cii_train_file_name)
    train.user_cat_data(train_path + params.ciii_train_file_name)

def test_csv(month=[7], save_pos='/7'):
    test = Feature(month=month, train=False, save_pos=save_pos)
    test_path = params.test_path + save_pos
    print(test_path)
    test.user_amt_data(test_path + params.civ_test_file_name)
    test.user_data(test_path + params.ci_test_file_name)
    test.user_brand_data(test_path + params.cii_test_file_name)
    test.user_cat_data(test_path + params.ciii_test_file_name)
    test.user_item_data(test_path + params.b_test_file_name)

if __name__ == '__main__':
    # train_csv(month=[5, 6, 7], save_pos='/567')
    train_csv(month=[4, 5, 6], save_pos='/456')
    train_csv(month=[2, 3, 4], save_pos='/234')
    # test_csv(month=[5], save_pos='/5')
    # test_csv(month=[7], save_pos='/7')
