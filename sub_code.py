# -*- coding:utf-8 -*-
# import package
import pandas as pd
import numpy as np
import gc
import math
import time
import warnings
warnings._setoption('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('float_format', lambda x: '%.6f' % x)
pd.options.display.max_rows = None

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

# import data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
df = pd.concat([train, test])
event = pd.read_csv('dataset/event.csv')
df = pd.merge(df, event, on='event_id', how='left')
del train, test, event
gc.collect()

# feature engineering
print("**************************  basic feature  **************************")
df.rename(columns={'nhit': 'event_id_nhit',
                   'nhitreal': 'event_id_nhitreal',
                   'energymc': 'event_id_energymc',
                   'thetamc': 'event_id_thetamc',
                   'phimc': 'event_id_phimc',
                   'xcmc': 'event_id_xcmc',
                   'ycmc': 'event_id_ycmc'}, inplace=True)

def cal_cos(x):
    return np.cos(x)

def cal_sin(x):
    return np.sin(x)

df['one'] = 1
df['cumsum'] = df.groupby(['event_id'])['one'].cumsum()
df['t1'] = df['t'].apply(lambda x: x if x < 2000 else 2000)
df.sort_values(by=['event_id', 't1'], inplace=True)
df['t_cumsum'] = df.groupby(['event_id'])['one'].cumsum()
df['t_slope'] = df['t1'] / df['t_cumsum']
df['x_slope'] = df['t1'] / df['x']
df['ty'] = df['y'] - df['t1']
df['tx'] = df['x'] - df['t_cumsum']
df['t_dis'] = np.sqrt(df['ty']**2 + df['tx']**2)
df['gap_cumsum'] = df['cumsum'] - df['t_cumsum']
df['tlabel'] = LabelEncoder().fit_transform(df['t1'])
df.sort_values(['event_id', 'tlabel'], inplace=True)
for i in range(16, 25):
    df['t_diff_{}'.format(i)] = df.groupby('event_id')[
        'tlabel'].diff(periods=i).fillna(method='bfill')
df.sort_values(['event_id', 'cumsum'], inplace=True)
for i in range(8, 22, 2):
    df['hitid_diff_{}'.format(i)] = df.groupby('event_id')[
        't1'].diff(periods=i).fillna(method='bfill')

slope, intercept = -0.38253218, 7.73506202
df['point_line'] = df.apply(
    lambda row: math.fabs(
        slope *
        row['x'] -
        row['t'] +
        intercept) /
    math.pow(
        slope**2 +
        1,
        0.5),
    axis=1)


def group_fea(df, target):
    df['event_id_{}_min'.format(target)] = df.groupby(
        'event_id')[target].transform('min')
    df['event_id_{}_max'.format(target)] = df.groupby(
        'event_id')[target].transform('max')
    df['event_id_{}_median'.format(target)] = df.groupby('event_id')[
        target].transform('median')
    df['event_id_{}_mean'.format(target)] = df.groupby('event_id')[
        target].transform('mean')
    df['event_id_{}_std'.format(target)] = df.groupby(
        'event_id')[target].transform('std')


print("**************************  统计特征  **************************")
# x, y, t,q 统计特征
group_fea(df, 'x')
group_fea(df, 'y')
group_fea(df, 'q')
group_fea(df, 't')

df['event_id_thetamc_cos'] = df['event_id_thetamc'].apply(lambda x: cal_cos(x))
df['event_id_thetamc_sin'] = df['event_id_thetamc'].apply(lambda x: cal_sin(x))
df['event_id_phimc_sin'] = df['event_id_phimc'].apply(lambda x: cal_sin(x))
df['event_id_phimc_cos'] = df['event_id_phimc'].apply(lambda x: cal_cos(x))
df['l'] = df['event_id_thetamc_sin'] * df['event_id_phimc_cos']
df['lx'] = df['l'] * df['x']
df['m'] = df['event_id_thetamc_sin'] * df['event_id_phimc_sin']
df['my'] = df['m'] * df['y']
df['h'] = (df['lx'] + df['my'] + 29.98 * df['event_id_t_mean']) / 29.98
df['t_h'] = df['t'] - df['h']
df['t_h_2'] = (df['t'] - df['h']) ** 2
group_fea(df, 't_h_2')

print("**************************  diff feature **************************")


def diff_stat(target):
    df['{}_min_diff'.format(target)] = df[target] - \
        df['event_id_{}_min'.format(target)]
    df['{}_max_diff'.format(target)] = df['event_id_{}_max'.format(
        target)] - df[target]
    df['{}_median_diff'.format(
        target)] = df['event_id_{}_median'.format(target)] - df[target]
    df['{}_mean_diff'.format(
        target)] = df['event_id_{}_mean'.format(target)] - df[target]


# # x,y,t,q "偏移"
diff_stat('x')
diff_stat('y')
diff_stat('t')
diff_stat('q')
diff_stat('t_h_2')

print("************************** time diff  feature **************************")
# t_h_2特征diff，强特
df = df.sort_values(by=['event_id', 't_h_2_min_diff']).reset_index(drop=True)
for i in range(10, 18):
    df['t_h_2_diff_last_{}'.format(i)] = df.groupby(['event_id'])[
        't_h_2'].diff(periods=i).fillna(0)

# 时间变化特征, 强特
df = df.sort_values(by=['event_id', 't_min_diff']).reset_index(drop=True)
for i in range(4, 16):
    df['t_diff_last_{}'.format(i)] = df.groupby('event_id')[
        't'].diff(periods=i).fillna(0)

# 修正时间, 没太大作用
df['t_minus_terror'] = df['t'] - df['terror']
df['t_add_terror'] = df['t'] + df['terror']
df['x_t'] = df['x'] / df['t_min_diff']

df['x_div_xcmc'] = df['x'] / (df['event_id_xcmc'] + 0.01)
df['y_div_ycmc'] = df['y'] / (df['event_id_ycmc'] + 0.01)
df['x_minus_xcmc'] = np.abs(df['x'] - df['event_id_xcmc'])
df['y_minus_ycmc'] = np.abs(df['y'] - df['event_id_ycmc'])
df['x_minus_xcmc_2'] = (df['x'] - df['event_id_xcmc'])**2
df['y_minus_ycmc_2'] = (df['y'] - df['event_id_ycmc'])**2

print("************************ position fearture **************************")
group_fea(df, 'x_minus_xcmc_2')
diff_stat('x_minus_xcmc_2')
df = df.sort_values(
    by=['event_id', 'x_minus_xcmc_2_min_diff']).reset_index(drop=True)
for i in range(4, 15):
    df['x_minus_xcmc_2_diff_last_{}'.format(i)] = df.groupby(
        ['event_id'])['x_minus_xcmc_2'].diff(periods=i).fillna(0)

# sort x, y
df = df.sort_values(by=['event_id', 'x', 'y']).reset_index(drop=True)
for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    df['x_y_diff_last_{}'.format(i)] = df.groupby(
        'event_id')['t'].diff(periods=i).fillna(0)

# 位置的变化特征, 线上 +3 左右
for i in range(5, 18):
    df['x_diff_last_{}'.format(i)] = df.groupby(['event_id'])[
        'x'].diff(periods=i).fillna(0)
for i in range(1, 10):
    df['y_diff_last_{}'.format(i)] = df.groupby(['event_id'])[
        'y'].diff(periods=i).fillna(0)

# 与中心距离的位置变化特征, 线上 +1 左右
df['dis2c'] = ((df['x'] - df['event_id_xcmc'])**2 +
               (df['y'] - df['event_id_ycmc'])**2)**0.5
group_fea(df, 'dis2c')
diff_stat('dis2c')
df = df.sort_values(by=['event_id', 'dis2c', 'x']).reset_index(drop=True)
for i in range(10, 15):
    df['dis2c_diff_last_{}'.format(i)] = df.groupby(['event_id'])[
        'dis2c'].diff(periods=i).fillna(0)

print("************************ others fearture **************************")
# 这个特征是比较有用的 event_id 特征
df['event_id_realhit_ratio'] = df['event_id_nhitreal'] / df['event_id_nhit']
df['event_id_fakehit_ratio'] = 1 - \
    df['event_id_nhitreal'] / df['event_id_nhit']
df['event_id_energymc_hit_ratio'] = df['event_id_energymc'] / df['event_id_nhit']

# q,t 变换
df['q2'] = df['q']**2
df['t2'] = df['t']**2
df['logt'] = np.log(df['t'])
df['logq'] = np.log(df['q'])
df['expq'] = np.exp(df['q'])

df['slope'] = df['y'] / df['x']
df['slope_mc'] = (df['y'] - df['event_id_ycmc']) / \
    (df['x'] - df['event_id_xcmc'])
df['angle_diff'] = df['slope'] - df['event_id_phimc']
df['area'] = np.abs(df['x'] * df['y'])

df['dis'] = np.sqrt(df['x']**2 + df['y']**2)
df['dis_mc'] = np.sqrt(df['event_id_xcmc']**2 + df['event_id_ycmc']**2)
df['dis_b'] = df['dis'] - df['dis_mc']

df['x2'] = df['x']**2
df['y2'] = df['y']**2

df['x_cmc'] = (df['x'] - df['event_id_xcmc']) / \
    (df['event_id_xcmc'] + df['event_id_xcmc'].mean())
df['y_cmc'] = (df['y'] - df['event_id_ycmc']) / \
    (df['event_id_ycmc'] + df['event_id_ycmc'].mean())
df['nhit_bias'] = (df['event_id_nhit'] - df['event_id_nhitreal']) / \
    (df['event_id_nhitreal'] + df['event_id_nhitreal'].mean())

# freq encoding: 没太大作用, 线上 +0.1


def freq_enc(df, col):
    vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
    df['{}_freq'.format(col)] = df[col].map(vc)
    return df


df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')
df = freq_enc(df, 'terror')
df = freq_enc(df, 'x_y')

id_and_label = ['event_id', 'hit_id', 'flag']
useless_features = [
    'z',
    'x_y',
    'event_id_t_min',
    'event_id_t_max',
    'event_id_nhit',
    'event_id_y_min',
    'event_id_x_min',
    'event_id_y_max',
    'event_id_x_max',
    'event_id_q_std',
    'event_id_q_max',
    'event_id_y_median',
    'event_id_x_median',
    'x_max_diff',
    'y_max_diff',
    'dis_mc',
    'event_id_terror_min',
    'one',
    't1',
    'tlabel',
    'l',
    'lx',
    'm',
    'my',
    't_h',
    'event_id_thetamc_cos',
    'event_id_thetamc_sin',
    'event_id_phimc_cos',
    'event_id_phimc_sin',
    'event_id_x',
    'event_id_x_minus_xcmc_2_std',
    'event_id_x_minus_xcmc_2_max',
    'event_id_x_minus_xcmc_2_mean',
    'event_id_t_h_2_std',
    'event_id_dis2c_max',
    'one']

use_features = [
    col for col in df.columns if col not in id_and_label +
    useless_features]

# 伪标签, 还算有点用, 线上 +0.5
# t < -900 ==> 0
# t > 1850 ==> 1  || q < 0    ==> 1
test = df[df.flag.isna()]
df.loc[df.flag.isna() & (df.t < -900), 'flag'] = 0
df.loc[df.flag.isna() & ((df.t > 1850) | (df.q < 0)), 'flag'] = 1

train = df[df.flag.notna()]
train['flag'] = train['flag'].astype('int')

del df
gc.collect()

print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  build lgb and training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
lgb_index = 9


def run_lgb(df_train, df_test, use_features, params):
    target = 'flag'
    oof_pred = np.zeros((len(df_train),))
    y_pred = np.zeros((len(df_test),))

    folds = GroupKFold(n_splits=6)
    for fold, (tr_ind, val_ind) in enumerate(
            folds.split(train, train[target], train['event_id'])):
        start_time = time.time()
        print('Fold {}'.format(fold + 1))
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        model = lgb.train(params,
                          train_set,
                          num_boost_round=5000,
                          early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features],
                                model.best_iteration) / folds.n_splits
        df_test['lgb_pred'] = model.predict(
            df_test[use_features], model.best_iteration)

        print("Features importance...")
        gain = model.feature_importance('gain')
        feat_imp = pd.DataFrame({'feature': model.feature_name(), 'split': model.feature_importance(
            'split'), 'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        print(feat_imp)

        used_time = (time.time() - start_time) / 3600
        print('used_time: {:.2f} hours'.format(used_time))

        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
    return oof_pred, y_pred

# def run_lgb_single(df_train, df_test, use_features, params):
#     target = 'flag'
#     start_time = time.time()
#     x_train, x_val, y_train, y_val = train_test_split(
#         df_train[use_features], df_train[target], test_size=0.2, random_state=0)
#     train_set = lgb.Dataset(x_train, y_train)
#     val_set = lgb.Dataset(x_val, y_val)

#     model = lgb.train(params,
#                       train_set,
#                       num_boost_round=5000,
#                       early_stopping_rounds=100,
#                       valid_sets=[train_set, val_set],
#                       verbose_eval=100)

#     oof_pred = model.predict(df_train[use_features])
#     y_pred = model.predict(df_test[use_features])

#     print("Features importance...")
#     gain = model.feature_importance('gain')
#     feat_imp = pd.DataFrame({'feature': model.feature_name(), 'split': model.feature_importance(
#         'split'), 'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
#     print(feat_imp)
#     used_time = (time.time() - start_time) / 3600
#     print('used_time: {:.2f} hours'.format(used_time))

#     del x_train, x_val, y_train, train_set, val_set
#     gc.collect()
#     return y_pred, oof_pred, y_val


params = {
    'learning_rate': 0.1,
    'metric': 'auc',
    'objective': 'binary',
    'feature_fraction': 0.80,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'n_jobs': -1,
    'seed': 2019,
    'max_depth': 8,
    'num_leaves': 64,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'boost_from_average': False
}

# # single
# y_pred, oof_pred, y_val = run_lgb_single(train, test, use_features,params)
# score = roc_auc_score(train['flag'], oof_pred)

oof_pred, y_pred = run_lgb(train, test, use_features, params)
score = roc_auc_score(train['flag'], oof_pred)
print("auc score:", score)


def eval(row, margin_num):
    if (row['nhitreal'] / row['nhit']) > 0.47 and row['n'] <= (row['nhitreal'] +
                                                               margin_num - int((row['nhitreal'] / row['nhit'] - 0.47) * 7)):
        return 1
    elif (row['nhitreal'] / row['nhit']) <= 0.47 and row['n'] <= (
            row['nhitreal'] + margin_num + int((0.47 - row['nhitreal'] / row['nhit']) * 10)):
        return 1
    else:
        return 0


def write_to_csv(df_test, df_event, margin_num=8):
    """ use nhitreal to improve recall
        df_test(pandas dataframe): event_id、 hit_id、flag_pred
        event_df(pandas dataframe): event_id、hit_id、nhitreal
    """
    df_test = pd.merge(df_test, df_event, on='event_id', how='left')
    df_test = df_test.sort_values(['event_id', 'flag_pred'], ascending=False)

    df_test['one'] = 1
    df_test['n'] = df_test.groupby(['event_id'])['one'].cumsum()
    df_test['flag_pred'] = df_test.apply(
        lambda row: eval(row, margin_num), axis=1)
    df_test = df_test.sort_values(by='hit_id')
    df_test[['hit_id', 'flag_pred', 'event_id']].to_csv(
        'sub_nhit_{}.csv'.format(margin_num), index=False)


test['flag_pred'] = y_pred
train['flag_pred'] = oof_pred
train[['hit_id', 'flag_pred', 'event_id']].to_csv(
    'train_lgb_{}_{}.csv'.format(score, str(lgb_index)), index=False)
test[['hit_id', 'flag_pred', 'event_id']].to_csv(
    'test_lgb_{}_{}.csv'.format(score, str(lgb_index)), index=False)

# lgb产生预测结果
for i in [2, 3, 4]:
    write_to_csv(test[['hit_id', 'flag_pred', 'event_id']],
                 event, margin_num=i)
    print(i)

# print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  level 1 build xgb and train >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#
#
#
# xgb_index = 8
#
#
# def run_xgb(df_train, df_test, use_features):
#     oof_pred = np.zeros((len(df_train),))
#     y_pred = np.zeros((len(df_test),))
#
#     folds = GroupKFold(n_splits=6)
#     for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['flag'], train['event_id'])):
#         start_time = time.time()
#         print('Fold {}'.format(fold + 1))
#
#         x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
#         y_train, y_val = df_train['flag'].iloc[tr_ind], df_train['flag'].iloc[val_ind]
#
#         clf = xgb.XGBClassifier(tree_method='hist', max_depth=8, learning_rate=0.1, verbosity=1,
#                                 eval_metric='auc', n_estimators=6000,nthread=10)
#         clf.fit(x_train, y_train,
#                 eval_set=[(x_train, y_train), (x_val, y_val)],
#                 early_stopping_rounds=100,
#                 verbose=100,
#                 )
#         oof_pred[val_ind] = clf.predict(x_val)
#         y_pred += clf.predict(df_test[use_features], clf.best_iteration) / folds.n_splits
#
#         used_time = (time.time() - start_time) / 3600
#         print('used_time: {:.2f} hours'.format(used_time))
#
#         del x_train, x_val, y_train, y_val
#         gc.collect()
#
#     return oof_pred, y_pred
#
#
# oof_pred, y_pred = run_xgb(train, test, use_features)
# score = roc_auc_score(train['flag'], oof_pred)
#
# test['xgb_pred'] = y_pred
# train['xgb_pred'] = oof_pred
# train[['hit_id', 'xgb_pred', 'event_id']].to_csv('train_xgb_{}.csv'.format(str(xgb_index)), index=False)
# test[['hit_id', 'xgb_pred', 'event_id']].to_csv('test_xgb_{}.csv'.format(str(xgb_index)), index=False)
#
#
# print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  level 1 ensemble >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

#
# def model_ensemble(df, event, ensemble_methods="avg",wts=None):
#     """ df(datafraem): col: eventid, hitid, pred1, pred2, pred3, ...
#         event(pandas dataframe): origin data read by pandas
#     """
#     use_feature = [fea for fea in df.columns.values if fea not in ["hit_id", "event_id"]]
#     if ensemble_methods == "avg":
#         df['flag_pred'] = np.mean(df[use_feature], axis=1)
#
#     if ensemble_methods == "weight" and len(wts) == len(use_feature):
#         df['flag_pred'] = np.average(df[use_feature], axis=1, weights=wts)
#
#     if ensemble_methods == "geomean":
#         pass
#
#     if ensemble_methods == "vote":
#         pass
#     for i in [3,4,5,6]:
#         write_to_csv(df,event,i)
#
#
# train_lgb_1 = pd.read_csv('test_xgb_{}.csv'.format(xgb_index))
# train_xgb_1 = pd.read_csv('test_lgb_{}.csv'.format(lgb_index))
# df_level1_output = pd.merge(train_lgb_1, train_xgb_1, on=['event_id', 'hit_id'], how='left')
# event = pd.read_csv('dataset/event.csv')
# model_ensemble(df_level1_output, event, ensemble_methods="avg", wts=None)
