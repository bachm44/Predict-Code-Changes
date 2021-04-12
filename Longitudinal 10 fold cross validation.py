import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, f1_score
import numpy as np
import time
from config import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# to suppress convergence warning in LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def main():
    feature_list = initial_feature_list

    print(project)
    df = pd.read_csv(f'{root}/{project}.csv')
    df_copy = df.copy()

    # scaling
    scaler = StandardScaler()
    df[feature_list] = scaler.fit_transform(df[feature_list])
    # selecting_classifier(df, feature_list)
    # hyper_tuning(df, feature_list)

    # this method does new author and effectiveness result also.
    cross_validation(df, df_copy, scaler)

    ## dimension wise results
    # dimension_validation(df)

    ## multiple revisions
    # multiple_revisions()


# achieved after hypertuning
n_estimators = 500
learning_rate = 0.01

def get_model():
    return LGBMClassifier(class_weight='balanced', subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))

def get_best_model():
    return LGBMClassifier(class_weight='balanced', n_estimators=n_estimators, learning_rate=learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))

def selecting_classifier(df, feature_list):
    models = [
        RandomForestClassifier(class_weight='balanced'),
        GradientBoostingClassifier(), # class weight parameter does not apply for gradient boosting
        ExtraTreesClassifier(class_weight='balanced'),
        LogisticRegression(class_weight='balanced', solver='saga'), # solver='saga' needed for randomness
        LGBMClassifier(class_weight='balanced')
    ]

    for model in models:
        print(model.__class__.__name__)
        total_result = None
        for run in range(runs):
            result = Result()
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

                x_train, y_train = df.loc[:train_size - 1, feature_list], \
                                   df.loc[:train_size - 1, target]
                x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                                 df.loc[train_size:test_size - 1, target]

                # because LightGBM does not sample data by default, hence would have given same results in each run
                if model.__class__.__name__ == 'LGBMClassifier': clf = get_model()
                else: clf = model
                clf.fit(x_train, y_train)

                y_prob = clf.predict_proba(x_test)[:, 1]
                result.calculate_result(y_test, y_prob, fold, False)

            result_df = result.get_df()
            if run:
                total_result += result_df
            else: total_result = result_df

        total_result /= runs
        total_result = total_result.mean()
        print(total_result[['auc', 'f1_score_m', 'f1_score_a']].values)
        print()

    print("UnderSampling and OverSampling train data")
    # parameters are not set as train data is randomly sampled
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        ExtraTreesClassifier(),
        LogisticRegression(),
        LGBMClassifier()
    ]

    underSampler = RandomUnderSampler()
    overSampler = RandomOverSampler()
    for model in models:
        print(model.__class__.__name__)
        total_under_result = total_over_result = None
        for run in range(runs):
            under_result, over_result = Result(), Result()
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

                x_train, y_train = df.loc[:train_size - 1, feature_list], \
                                   df.loc[:train_size - 1, target]

                x_train_under, y_train_under = underSampler.fit_resample(x_train, y_train)
                x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                                 df.loc[train_size:test_size - 1, target]

                clf = model
                clf.fit(x_train_under, y_train_under)

                y_prob = clf.predict_proba(x_test)[:, 1]
                under_result.calculate_result(y_test, y_prob, fold, False)

                x_train_over, y_train_over = overSampler.fit_resample(x_train, y_train)
                clf = model
                clf.fit(x_train_over, y_train_over)
                y_prob = clf.predict_proba(x_test)[:, 1]
                over_result.calculate_result(y_test, y_prob, fold, False)

            result_df = under_result.get_df()
            if run: total_under_result += result_df
            else: total_under_result = result_df

            result_df = over_result.get_df()
            if run: total_over_result += result_df
            else: total_over_result = result_df

        total_under_result /= runs
        total_over_result /= runs

        total_under_result, total_over_result = total_under_result.mean(), total_over_result.mean()
        print(total_under_result[['auc', 'f1_score_m', 'f1_score_a']].values, total_over_result[['auc', 'f1_score_m', 'f1_score_a']].values)
        print()


def hyper_tuning(df, feature_list):
    total_result = None
    for n_estimators in [100, 500]:
        for learning_rate in [0.1, 0.01]:
                print(n_estimators, learning_rate)
                for run in range(runs):
                    result = Result()
                    for fold in range(1, folds):
                        train_size = df.shape[0] * fold // folds
                        test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

                        x_train, y_train = df.loc[:train_size - 1, feature_list], \
                                           df.loc[:train_size - 1, target]
                        x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                                         df.loc[train_size:test_size - 1, target]

                        clf = get_model()
                        clf.fit(x_train, y_train)

                        y_prob = clf.predict_proba(x_test)[:, 1]
                        result.calculate_result(y_test, y_prob, fold, False)

                    result_df = result.get_df()
                    if run:
                        total_result += result_df
                    else:
                        total_result = result_df

                total_result /= runs
                total_result = total_result.mean()
                print(total_result[['auc', 'f1_score_m', 'f1_score_a']].values)
                print()


def multiple_revisions():
    print("Multiple revisions")
    total_df = pd.read_csv(f'{root}/{project}_multiple_revisions.csv')
    total_df['last_revision_no'] = total_df.groupby('change_id')['number_of_revision'].transform('max').astype(int)

    total_df = total_df.sort_values(by=['created'], ascending=True).reset_index(drop=True)

    feature_list = initial_feature_list

    total_df_copy = total_df.copy()
    total_df[feature_list] = StandardScaler().fit_transform(total_df[feature_list])

    print("Approach 1: result adding only revision no")
    feature_list = feature_list + ['number_of_revision']

    for run in range(runs):
        test_result = Result()
        results = [Result() for _ in range(2)]

        for fold in range(1, folds):
            train_size = total_df.shape[0] * fold // folds
            test_size = min(total_df.shape[0] * (fold + 1) // folds, total_df.shape[0])

            x_train, y_train = total_df.loc[:train_size - 1, feature_list], \
                               total_df.loc[:train_size - 1, target]
            x_test, y_test = total_df.loc[train_size:test_size - 1, feature_list], \
                             total_df.loc[train_size:test_size - 1, target]

            clf = get_best_model()
            clf.fit(x_train, y_train)

            y_prob = clf.predict_proba(x_test)[:, 1]
            test_result.calculate_result(y_test, y_prob, fold, False)

            revision_no = total_df_copy.loc[train_size:test_size - 1, 'number_of_revision']
            for i, result in enumerate(results):
                if i + 1 == len(results):
                    index = revision_no == total_df_copy.loc[train_size:test_size - 1, 'last_revision_no']
                else:
                    index = revision_no == i + 1
                y_prob = clf.predict_proba(x_test[index])[:, 1]
                result.calculate_result(y_test[index], y_prob, fold, False)

        test_result_df = test_result.get_df()
        print(test_result_df.mean())

        for result in results:
            print(result.get_df().mean())
        print()
        break

    print("Approach2 : result using history of prior patches")
    feature_list = initial_feature_list + late_features
    for run in range(runs):
        test_result = Result()
        results = [Result() for _ in range(2)]

        for fold in range(1, folds):
            train_size = total_df.shape[0] * fold // folds
            test_size = min(total_df.shape[0] * (fold + 1) // folds, total_df.shape[0])

            x_train, y_train = total_df.loc[:train_size - 1, feature_list], \
                               total_df.loc[:train_size - 1, target]
            x_test, y_test = total_df.loc[train_size:test_size - 1, feature_list], \
                             total_df.loc[train_size:test_size - 1, target]

            clf = get_best_model()
            clf.fit(x_train, y_train)

            y_prob = clf.predict_proba(x_test)[:, 1]
            test_result.calculate_result(y_test, y_prob, fold, False)

            revision_no = total_df_copy.loc[train_size:test_size - 1, 'number_of_revision']
            for i, result in enumerate(results):
                if i + 1 == len(results):
                    index = revision_no == total_df_copy.loc[train_size:test_size - 1, 'last_revision_no']
                else:
                    index = revision_no == i + 1
                y_prob = clf.predict_proba(x_test[index])[:, 1]
                result.calculate_result(y_test[index], y_prob, fold, False)

        test_result_df = test_result.get_df()
        print(test_result_df.mean())

        for result in results:
            print(result.get_df().mean())
        print()
        break


def cross_validation(df, df_copy, scaler):
    print("Cross validation")
    train_results = None
    test_results = None
    new_author_results = None
    feature_importances = []
    train_time = {fold: [] for fold in range(1, folds)}
    test_time = {fold: [] for fold in range(1, folds)}
    feature_list = get_initial_feature_list()
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
            clf = get_best_model()
            clf.fit(x_train, y_train)
            train_time[fold].append(time.time() - start)

            y_prob = clf.predict_proba(x_train)[:, 1]
            train_result.calculate_result(y_train, y_prob, fold, False)

            start = time.time()
            y_prob = clf.predict_proba(x_test)[:, 1]
            test_time[fold].append(time.time() - start)
            test_result.calculate_result(y_test, y_prob, fold, False)

            for k in range(1, 10):
                score = Result.cost_effectiveness(y_test, y_prob, k*10)
                scores[k-1] += score

            test_new = df_copy[train_size:test_size]
            test_new = test_new[test_new['total_change_num'] < 10]
            y_prob = clf.predict_proba(scaler.transform(test_new[feature_list]))[:, 1]
            new_author_result.calculate_result(test_new[target], y_prob, fold, False)

            feature_importances.append(clf.feature_importances_)

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
                              "new_developers":new_author_results.mean()}).reset_index()
    print(result_df)

    print("Effectiveness")
    for k in range(1, 10):
        print(k*10, scores[k-1] / (runs*(folds - 1)))

    # average time
    for fold in range(1, folds):
        train_time[fold] = np.mean(train_time[fold])
        test_time[fold] = np.mean(test_time[fold])
    train_results['time'] = train_time.values()
    test_results['time'] = test_time.values()

    train_results.to_csv(f'{root}/{project}_train_result_cross.csv', index=False, float_format='%.3f')
    test_results.to_csv(f'{root}/{project}_test_result_cross.csv', index=False, float_format='%.3f')
    result_df.to_csv(f'{root}/{project}_result_cross.csv', index=False, float_format='%.3f')

    # process and dump feature importance
    feature_importance_df = pd.DataFrame({'feature': feature_list, 'importance': np.mean(feature_importances, axis=0)})
    feature_importance_df['importance'] = feature_importance_df['importance'] * 100 / feature_importance_df[
        'importance'].sum()
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(f'{root}/{project}_feature_importance_cross.csv', index=False, float_format='%.3f')
    print()


def dimension_validation(df):
    print("Varying dimensions")

    for group in features_group:
        print(group)
        feature_sub_list = features_group[group]
        total_result = None

        for run in range(runs):
            result = Result()
            for fold in range(1, folds):
                train_size = df.shape[0] * fold // folds
                test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

                x_train, y_train = df.loc[:train_size - 1, feature_sub_list], df.loc[:train_size - 1,target]
                x_test, y_test = df.loc[train_size:test_size - 1, feature_sub_list], \
                                 df.loc[train_size:test_size - 1,target]

                clf = get_best_model()
                clf.fit(x_train, y_train)

                y_prob = clf.predict_proba(x_test)[:, 1]
                result.calculate_result(y_test, y_prob, fold, False)

            result_df = result.get_df()
            if run: total_result+= result_df
            else: total_result = result_df

        total_result /= runs
        print(total_result.mean())
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