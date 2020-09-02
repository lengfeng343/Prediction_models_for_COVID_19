from xgboost import plot_tree
from Prediction_models_for_COVID_19.utils import StratifiedKFold_func_with_features_sel
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from numpy import *
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


class Predict:
    def __init__(self, train_file, test_file):
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

        train_df = train_df.drop(['PATIENT_ID', '性别', '年龄'], axis=1)
        test_df = test_df.drop(['PATIENT_ID', '性别', '年龄'], axis=1)

        cols = list(train_df.columns)
        cols.remove('出院方式')
        self.x_train = train_df[cols]
        self.y_train = train_df['出院方式']

        cols = list(test_df.columns)
        cols.remove('出院方式')
        self.x_test = test_df[cols]
        self.y_test = test_df['出院方式']

        print('training set:', len(self.y_train), 'survival:', len([i for i in self.y_train if i == 0]), 'death:', len([i for i in self.y_train if i == 1]))
        print('testing set:', len(self.y_test), 'survival:', len([i for i in self.y_test if i == 0]), 'death:', len([i for i in self.y_test if i == 1]))

    def features_selection(self, method):
        """
        Select important features from all features
        :param method: Feature selection method
        :return: important features
        """
        X_data_all_features = self.x_train.copy()
        Y_data = self.y_train.copy()
        import_feature = pd.DataFrame()
        sel_cols = list(X_data_all_features.columns)
        import_feature['col'] = sel_cols
        import_feature['xgb'] = 0

        if method == 'LOGISTIC':
            model = SelectFromModel(LogisticRegression(penalty="l1", C=1.0))
            model.fit_transform(X_data_all_features, Y_data)
            cols = [list(sel_cols)[i] for i in range(len(model.get_support())) if model.get_support()[i]]
            print(cols)
            return cols

        if method == 'X2':
            model = SelectKBest(chi2, k=8)
            model.fit_transform(X_data_all_features.values, Y_data.values)
            pvalue = [1 for i in range(len(model.get_support())) if model.get_support()[i] and model.pvalues_[i] < 0.001]
            cols = [list(sel_cols)[i] for i in range(len(model.get_support())) if model.get_support()[i]]
            print('X2 Selected features:', cols)
            return cols

        if method == 'XGBoost':
            for i in range(100):
                x_train, x_test, y_train, y_test = train_test_split(X_data_all_features, Y_data, test_size=0.3, random_state=i)
                model = xgb.XGBClassifier(
                        max_depth=4,
                        learning_rate=0.2,
                        reg_lambda=1,
                        n_estimators=150,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        # scale_pos_weight=10,
                )
                model.fit(x_train, y_train)
                import_feature['xgb'] = import_feature['xgb']+model.feature_importances_/100
            import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
            print('Top 10 features:')
            print(import_feature.head(10))
            import_feature_cols = import_feature['col'].values[:10]

            # select features from top 10 features
            num_i = 1
            val_score_old = 0
            val_score_new = 0
            while val_score_new >= val_score_old:
                val_score_old = val_score_new
                x_col = import_feature_cols[:num_i]
                print(x_col)
                X_data = X_data_all_features[x_col]
                print('5-Fold CV:')
                acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
                print("Train AUC-score is %.4f ; Validation AUC-score is %.4f" % (acc_train,acc_val))
                print("Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f" % (acc_train_std,acc_val_std))
                val_score_new = acc_val
                num_i += 1

            print('Selected features:', x_col[:-1])
            return list(x_col[:-1])

        if method == 'RandomForest':
            for i in range(100):
                forest = RandomForestClassifier(n_estimators=150)
                forest.fit(X_data_all_features, Y_data)
                import_feature['xgb'] = import_feature['xgb']+forest.feature_importances_/100
            import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
            print('Top 10 features:')
            print(import_feature.head(10))
            import_feature_cols = import_feature['col'].values[:10]

            num_i = 1
            val_score_old = 0
            val_score_new = 0
            while val_score_new >= val_score_old:
                val_score_old = val_score_new
                x_col = import_feature_cols[:num_i]
                print(x_col)
                X_data = X_data_all_features[x_col]
                print('5-Fold CV:')
                acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
                print("Train AUC-score is %.4f ; Validation AUC-score is %.4f" % (acc_train,acc_val))
                print("Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f" % (acc_train_std,acc_val_std))
                val_score_new = acc_val
                num_i += 1

            print('Selected features:', x_col[:-1])
            return list(x_col[:-1])

        if method == 'ExtraTrees':
            import_feature = pd.DataFrame()
            import_feature['col'] = sel_cols
            model = ExtraTreesClassifier()
            model.fit(X_data_all_features, Y_data)
            import_feature['xgb'] = model.feature_importances_
            import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
            print('Top 10 features:')
            print(import_feature.head(10))
            import_feature_cols = import_feature['col'].values[:10]

            num_i = 1
            val_score_old = 0
            val_score_new = 0
            while val_score_new >= val_score_old:
                val_score_old = val_score_new
                x_col = import_feature_cols[:num_i]
                print(x_col)
                X_data = X_data_all_features[x_col]
                print('5-Fold CV:')
                acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
                print("Train AUC-score is %.4f ; Validation AUC-score is %.4f" % (acc_train,acc_val))
                print("Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f" % (acc_train_std,acc_val_std))
                val_score_new = acc_val
                num_i += 1

            print('Selected features:', x_col[:-1])
            return list(x_col[:-1])

        if method == 'RFE':
            model = LogisticRegression()
            rfe = RFE(model, 10)
            fit = rfe.fit(X_data_all_features, Y_data)
            cols = [list(sel_cols)[i] for i in range(len(fit.support_)) if fit.support_[i]]
            print('Selected features:', cols)
            return cols

        if method == 'GBDT':
            for i in range(100):
                x_train, x_test, y_train, y_test = train_test_split(X_data_all_features,Y_data,test_size=0.3,random_state=i)
                model = GradientBoostingClassifier(n_estimators=90, learning_rate=0.1, subsample=0.6, random_state=0)
                model.fit(x_train, y_train)
                import_feature['xgb'] = import_feature['xgb']+model.feature_importances_/100
            import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
            print('Top 10 features:')
            print(import_feature.head(10))
            import_feature_cols = import_feature['col'].values[:10]
            num_i = 1
            val_score_old = 0
            val_score_new = 0
            while val_score_new >= val_score_old:
                val_score_old = val_score_new
                x_col = import_feature_cols[:num_i]
                print(x_col)
                X_data = X_data_all_features[x_col]#.values
                print('5-Fold CV:')
                acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func_with_features_sel(X_data.values,Y_data.values)
                print("Train AUC-score is %.4f ; Validation AUC-score is %.4f" % (acc_train,acc_val))
                print("Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f" % (acc_train_std,acc_val_std))
                val_score_new = acc_val
                num_i += 1

            print('Selected features:', x_col[:-1])
            return list(x_col[:-1])

    def single_tree(self, cols, method):
        print('single_tree:')
        x_train = self.x_train[cols].values
        y_train = self.y_train.values

        x_test = self.x_test[cols].values
        y_test = self.y_test.values

        if method == 'XGBoost':
            model = xgb.XGBClassifier(
                max_depth=3,
                n_estimators=1,
            )
            model.fit(x_train, y_train)

            print('training set:')
            pred_train = model.predict(x_train)
            print(confusion_matrix(y_train, pred_train))
            print(classification_report(y_train, pred_train))

            print('testing set:')
            pred_test = model.predict(x_test)
            print(confusion_matrix(y_test, pred_test))
            print(classification_report(y_test, pred_test))

            plt.figure(dpi=300, figsize=(8, 6))
            plot_tree(model)  # draw the decision tree
            plt.show()

            graph = xgb.to_graphviz(model)
            graph.render(filename='single-tree.dot')

        if method == 'RandomForest':
            model = RandomForestClassifier(n_estimators=1)
            model.fit(x_train, y_train)

            print('training set:')
            pred_train = model.predict(x_train)
            print(confusion_matrix(y_train, pred_train))
            print(classification_report(y_train, pred_train))

            print('testing set:')
            pred_test = model.predict(x_test)
            print(confusion_matrix(y_test, pred_test))
            print(classification_report(y_test, pred_test))

        if method == 'GBDT':
            model = GradientBoostingClassifier(n_estimators=150,
                                               learning_rate=0.1)
            model.fit(x_train, y_train)
            print('training set:')
            pred_train = model.predict(x_train)
            print(confusion_matrix(y_train, pred_train))
            print(classification_report(y_train, pred_train))

            print('testing set:')
            pred_test = model.predict(x_test)
            print(confusion_matrix(y_test, pred_test))
            print(classification_report(y_test, pred_test))

        if method == 'LOGISTIC':
            model = linear_model.LogisticRegression(random_state=0, C=1, solver='lbfgs')
            model.fit(x_train, y_train)

            print('training set:')
            pred_train = model.predict(x_train)
            print(confusion_matrix(y_train, pred_train))
            print(classification_report(y_train, pred_train))

            print('testing set:')
            pred_test = model.predict(x_test)
            print(confusion_matrix(y_test, pred_test))
            print(classification_report(y_test, pred_test))

            b = model.intercept_
            s = ''
            for i in range(len(cols)):
                if model.coef_[0][i] > 0:
                    s += '+' + str(round(model.coef_[0][i],3)) + '*' + cols[i]
                else:
                    s += str(round(model.coef_[0][i], 3)) + '*' + cols[i]
            s += str(round(b[0], 3))
            print('Logistic regression equation:1/(1+exp-({}))'.format(s))

        pred_val_probe = model.predict_proba(x_test)[:, 1]
        plt.subplots(figsize=(7, 5.5))
        fpr, tpr, thresholds = roc_curve(y_test, pred_val_probe, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='Model (AUC = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        # plt.savefig('AUC_train.png')
        plt.show()


if __name__ == '__main__':
    model = Predict(train_file='../Pre_Surv_COVID_19/new data/train.xlsx', test_file='../Pre_Surv_COVID_19/new data/test.xlsx')
    cols = model.features_selection(method='GBDT')           # [LOGISTIC, X2, XGBoost, RandomForest, ExtraTrees, RFE, GBDT]
    # cols = ['乳酸脱氢酶', '淋巴细胞(%)', '超敏C反应蛋白']
    model.single_tree(cols, method='LOGISTIC')                  # [XGBoost, GBDT, RandomForest, LOGISTIC]
