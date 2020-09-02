from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve, auc


def StratifiedKFold_func_with_features_sel(x, y,Num_iter=100,score_type = 'auc'):
    acc_v = []
    acc_t = []
    for i in range(Num_iter):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x, y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]

            model = xgb.XGBClassifier(max_depth=4,learning_rate=0.2,reg_alpha=1)

            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)

            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_te, pred))
                acc_t.append(roc_auc_score(y_tr, train_pred))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


def survival_analysis(result_path):
    print('survival analysis...')
    df = pd.read_excel(result_path)

    kmf_0 = KaplanMeierFitter()
    kmf_1 = KaplanMeierFitter()

    group_0 = df['pred_m1'] == 0
    group_01 = df['pred_m1'] == 1

    logrank_test_res = logrank_test(
        df[group_0]['住院天数'], df[group_01]['住院天数'],
        df[group_0]['出院方式'], df[group_01]['出院方式'], alpha=.99)
    # if p_value<0.05，there is an obvious difference
    if logrank_test_res.p_value < 0.05:
        print(logrank_test_res.p_value, 'There is an obvious difference!')
    else:
        print(logrank_test_res.p_value, 'There is no obvious difference!')

    ax = plt.subplot(111)
    kmf_0.fit(df[group_0]['住院天数'], event_observed=df[group_0]['出院方式'], label=f'Predicted survival group')
    ax = kmf_0.plot(ax=ax)
    kmf_1.fit(df[group_01]['住院天数'], event_observed=df[group_01]['出院方式'], label=f'Predicted death group')
    ax = kmf_1.plot(ax=ax)
    plt.show()


def plt_ROC(result_path):
    df = pd.read_excel(result_path)
    y_true = df['出院方式']
    pred_b1 = df['pred_prob_b1']
    pred_b2 = df['pred_prob_b2']
    pred1 = df['pred_prob_m1']
    pred2 = df['pred_prob_m2']

    # Draw ROC curve
    plt.subplots(figsize=(7, 5.5))
    fpr, tpr, thresholds = roc_curve(y_true, pred_b1, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='red', lw=1, label='Benchmark-dt (AUC = %0.2f)' % roc_auc)

    fpr, tpr, thresholds = roc_curve(y_true, pred_b2, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='indigo', lw=1, label='Benchmark-lr  (AUC = %0.2f)' % roc_auc)

    fpr, tpr, thresholds = roc_curve(y_true, pred1, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='New-dt            (AUC = %0.2f)' % roc_auc)

    fpr, tpr, thresholds = roc_curve(y_true, pred2, pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkgreen', lw=1, label='New-lr             (AUC = %0.2f)' % roc_auc)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Reference line')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity', font2)
    plt.ylabel('Sensitivity', font2)
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    result_path = '../Pre_Surv_COVID_19/new data/ROC.xlsx'
    survival_analysis(result_path)
    plt_ROC(result_path)
