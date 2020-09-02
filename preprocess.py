import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def col_miss(df):
    col_missing_df = df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col', 'missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df


def is_number(s):
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def load_data(path):
    data_df = pd.read_excel(path, encoding='gbk', index_col=[0, 1])

    # select first appeared feature value
    data_df = (
        data_df
        .groupby(['PATIENT_ID']).bfill()
        .groupby(['PATIENT_ID']).first()
    )
    data_df = data_df.reset_index()

    data_df = data_df.drop(['入院时间', '出院时间'], axis=1)
    data_df = data_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df = data_df.applymap(lambda x: x if is_number(x) else -1)
    data_df = data_df.astype(float)
    return data_df


def select_features(data_df_unna):
    # calculate the lack of features
    col_miss_data = col_miss(data_df_unna)

    # calculate the proportion of missing features
    col_miss_data['Missing_part'] = col_miss_data['missing_count']/len(data_df_unna)
    # select features that are missing less than 0.25
    sel_cols = col_miss_data[col_miss_data['Missing_part'] <= 0.25]['col']
    print('select {} features'.format(len(sel_cols)))
    data_df_unna = data_df_unna[sel_cols]
    return data_df_unna


def missing_data_fill(df):
    cols = list(df.columns)
    df0 = df[df['出院方式'] == 0]
    df1 = df[df['出院方式'] == 1]
    for col in cols:
        if df0[col].isnull().any():
            # print(col, df0[col].mean())
            df0[col].fillna(df0[col].mean(), inplace=True)
        if df1[col].isnull().any():
            # print(col, df1[col].mean())
            df1[col].fillna(df1[col].mean(), inplace=True)
    df0 = df0.append(df1)
    return df0


def split_data(df, train_file, test_file):
    cols = list(df.columns)
    cols.remove('出院方式')
    X_data = df[cols]
    Y_data = df['出院方式']
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.5,
                                                                            random_state=1, stratify=Y_data)
    train_df = x_train.copy()
    train_df['出院方式'] = y_train
    test_df = x_test.copy()
    test_df['出院方式'] = y_test
    print('training set:', len(train_df), 'survival:', len([i for i in train_df['出院方式'] if i == 0]), 'death:',
          len([i for i in train_df['出院方式'] if i == 1]))
    print('testing set:', len(test_df), 'survival:', len([i for i in test_df['出院方式'] if i == 0]), 'death:',
          len([i for i in test_df['出院方式'] if i == 1]))
    train_df.to_excel(train_file, index=False)
    test_df.to_excel(test_file, index=False)


if __name__ == '__main__':
    filepath = '../Pre_Surv_COVID_19/new data/time_series_375_preprocess.xlsx'
    data_df = load_data(filepath)
    data_df = select_features(data_df)
    data_df = missing_data_fill(data_df)
    split_data(data_df, train_file='./test/train.xlsx', test_file='./test/test.xlsx')
