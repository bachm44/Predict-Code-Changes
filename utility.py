import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score
import numpy as np
import time

seed = 7
folds = 11
root = 'data'
target = 'status'

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num'],
    'text': ['description_length','is_documentation','is_bug_fixing','is_feature'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author'],
    'reviewer': ['reviewers_num', 'avg_reviewer_experience', 'avg_reviewer_review_count'],
    'code': ['lines_added','lines_deleted','files_added','files_deleted', 'files_modified','directory_num',
             'modify_entropy']
}


def get_initial_feature_list() -> [str]:
    feature_list = []
    for group in features_group:
        feature_list.extend(features_group[group])
    return feature_list


def load_features(project, revision_no=1):
    filepath = f'{root}/features/{project}.csv'
    feature_df = pd.read_csv(filepath)

    if revision_no is not None:
        feature_df = feature_df[feature_df['revision_no'] == revision_no]

    # selected_changes = pd.read_csv(f"{root}/selected changes/{project}_selected_changes.csv")[['change_id']]
    # feature_df = selected_changes.merge(feature_df, on=['change_id'], how='inner')
    feature_df = feature_df.sort_values(by=['created'], ascending=True).reset_index(drop=True)
    return feature_df


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