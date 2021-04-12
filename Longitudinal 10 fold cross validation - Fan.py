import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np
import time
from config import *


feature_list = ['change_num', 'recent_change_num', 'subsystem_change_num', 'review_num', 'merged_ratio',
     'recent_merged_ratio', 'subsystem_merged_ratio',

     'lines_added_num', 'lines_deleted_num', 'changed_file_num', 'files_added_num', 'files_deleted_num',
     'directory_num', 'subsystem_num', 'modify_entropy', 'language_num', 'file_type_num', 'segs_added_num',
     'segs_deleted_num', 'segs_updated_num',

     'changes_files_modified', 'file_developer_num',

     'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
     'eigenvector_centrality', 'clustering_coefficient', 'k_coreness',

     'msg_length', 'has_bug', 'has_feature', 'has_improve', 'has_document', 'has_refactor']


def main():
    print(project)
    df = pd.read_csv(f'{root}/{project}_fan_fixed.csv')
    train_results = None
    test_results = None
    new_author_results = None
    train_time = {fold: [] for fold in range(1, folds)}
    test_time = {fold: [] for fold in range(1, folds)}
    scores = [0] * 9

    for run in range(runs):
        test_result = Result()
        train_result = Result()
        new_author_result = Result()

        for fold in range(1, folds):
            train_size = df.shape[0] * fold // folds
            test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

            x_train, y_train = df.loc[:train_size - 1, feature_list], df.loc[:train_size - 1, target]
            x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                             df.loc[train_size:test_size - 1, target]

            start = time.time()
            clf = RandomForestClassifier(class_weight='balanced')
            clf.fit(x_train, y_train)
            train_time[fold].append(time.time() - start)

            y_prob = clf.predict_proba(x_train)[:, 1]
            train_result.calculate_result(y_train, y_prob, fold, False)

            start = time.time()
            y_prob = clf.predict_proba(x_test)[:, 1]
            test_time[fold].append(time.time() - start)
            test_result.calculate_result(y_test, y_prob, fold, False)

            for k in range(1, 10):
                score = Result.cost_effectiveness(y_test, y_prob, k * 10)
                scores[k - 1] += score

            train, test = df[:train_size], df[train_size:test_size]
            train, test = train[train['change_num'] < 10], test[test['change_num'] < 10]
            x_train, y_train = train[feature_list], train[target]
            x_test, y_test = test[feature_list], test[target]
            clf = RandomForestClassifier(class_weight='balanced')
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)[:, 1]
            new_author_result.calculate_result(y_test, y_prob, fold, False)

        train_result_df = train_result.get_df()
        test_result_df = test_result.get_df()
        new_author_result_df = new_author_result.get_df()

        if run:
            train_results += train_result_df
            test_results += test_result_df
            new_author_results += new_author_result_df
        else:
            train_results = train_result_df
            test_results = test_result_df
            new_author_results = new_author_result_df

    train_results /= runs
    test_results /= runs
    new_author_results /= runs

    result_df = pd.DataFrame({"train": train_results.mean(), "test": test_results.mean(),
                              "new_developers": new_author_results.mean()}).reset_index()
    print(result_df)

    print("Effectiveness")
    for k in range(1, 10):
        print(k * 10, scores[k - 1] / (runs * (folds - 1)))

    # average time
    for fold in range(1, folds):
        train_time[fold] = np.mean(train_time[fold])
        test_time[fold] = np.mean(test_time[fold])
    train_results['time'] = train_time.values()
    test_results['time'] = test_time.values()
    print(train_results['time'].mean())

    # train_results.to_csv(f'{root}/{project}_train_result_cross.csv', index=False, float_format='%.3f')
    # test_results.to_csv(f'{root}/{project}_test_result_cross.csv', index=False, float_format='%.3f')
    # result_df.to_csv(f'{root}/{project}_result_cross.csv', index=False, float_format='%.3f')
    print()


class Result:
    def __init__(self):
        self.folds = []
        self.auc = []
        self.accuracy = []
        self.effectiveness = []

        self.precision_m = []
        self.recall_m = []
        self.f1_score_m = []

        self.precision_a = []
        self.recall_a = []
        self.f1_score_a = []

    # evaluates the percentage of merged code changes over the top K%
    # suspicious merged code changes
    @staticmethod
    def cost_effectiveness(y_true, y_score, k):
        df = pd.DataFrame({'class': y_true, 'pred': y_score})
        df = df.sort_values(by=['pred'], ascending=False).reset_index(drop=True)
        if k > 100:
            print('K must be  > 0 and < 100')
            return -1
        df = df.iloc[:df.shape[0] * k // 100]

        merged_changes = df[df['class'] == 1].shape[0]
        changes = df.shape[0]

        if changes:
            return merged_changes / changes
        else:
            return 0

    def calculate_result(self, y_true, y_score, fold=None, verbose=False):
        if fold is not None:
            self.folds.append(fold)

        auc = roc_auc_score(y_true, y_score)
        cost_effectiveness = Result.cost_effectiveness(y_true, y_score, 20)
        if verbose: print(f'AUC {auc}, cost effectiveness {cost_effectiveness}.')
        self.auc.append(auc)

        self.effectiveness.append(cost_effectiveness)

        y_pred = np.round(y_score)

        self.accuracy.append(accuracy_score(y_true, y_pred))

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None)
        if verbose: print(f'precision {precision}, recall {recall}, f1_score {f1_score}.')
        self.precision_a.append(precision[0])
        self.recall_a.append(recall[0])
        self.f1_score_a.append(f1_score[0])

        self.precision_m.append(precision[1])
        self.recall_m.append(recall[1])
        self.f1_score_m.append(f1_score[1])

    def get_df(self):
        return pd.DataFrame({
            'fold': self.folds,
            'auc': self.auc,
            'accuracy': self.accuracy,
            'cost_effectiveness': self.effectiveness,
            'precision_m': self.precision_m,
            'recall_m': self.recall_m,
            'f1_score_m': self.f1_score_m,
            'precision_a': self.precision_a,
            'recall_a': self.recall_a,
            'f1_score_a': self.f1_score_a,
        })

    def process(self, number):
        return np.round(np.mean(number), 2)

    def show(self):
        print(
            f"{self.process(self.auc)} & {self.process(self.effectiveness)} & {self.process(self.f1_score_m)} & {self.process(self.precision_m)} & {self.process(self.recall_m)} & {self.process(self.f1_score_a)} & {self.process(self.precision_a)} & {self.process(self.recall_a)}")


if __name__ == '__main__':
    main()