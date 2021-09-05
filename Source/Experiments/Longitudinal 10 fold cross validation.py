from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from ..Util import *

# to suppress convergence warning in LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# achieved after hypertuning
best_n_estimators = 500
best_learning_rate = 0.01


def main():
    feature_list = initial_feature_list

    print(project)
    df = pd.read_csv(f'{root}/{project}.csv')
    df_copy = df.copy()

    # scaling
    scaler = StandardScaler()
    df[feature_list] = scaler.fit_transform(df[feature_list])
    # selecting_classifier(df, feature_list)

    # this method does new author and effectiveness result also.
    cross_validation(df, df_copy, scaler)

    ## dimension wise results
    # dimension_validation(df)

    ## multiple revisions
    # multiple_revisions()


def get_model():
    return LGBMClassifier(class_weight='balanced', subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


def get_best_model():
    return LGBMClassifier(class_weight='balanced', n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


def selecting_classifier(df):
    print("Selecting best classifier with AUC, F1(M) and F1(A)")
    print("RandomForest")
    for n_estimators in [100, 500]:
        for max_depth in [None, 10, 15]:
            print(n_estimators, max_depth)
            run_model(RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth), df)

    print("GradientBoost")
    for n_estimators in [100, 500]:
        for learning_rate in [0.1, 0.01]:
            print(n_estimators, learning_rate)
            run_model(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate), df)

    print("ExtraTrees")
    for n_estimators in [100, 500]:
        for max_depth in [None, 5, 10]:
            print(n_estimators, max_depth)
            run_model(ExtraTreesClassifier(class_weight='balanced', n_estimators=n_estimators, max_depth=max_depth), df)

    print("LogisticRegression")
    for max_iter in [50, 100, 500]:
        print(max_iter)
        run_model(LogisticRegression(class_weight='balanced', solver='saga', max_iter=max_iter), df)

    print("LightGBM")
    for n_estimators in [100, 500]:
        for learning_rate in [0.1, 0.01]:
                print(n_estimators, learning_rate)
                model = LGBMClassifier(class_weight='balanced', n_estimators=n_estimators, random_state=np.random.randint(seed),
                                     learning_rate=learning_rate, subsample=0.9, subsample_freq=1)
                run_model(model, df)


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

    train_results.to_csv(f'{result_project_folder}/{project}_train_result_cross.csv', index=False, float_format='%.3f')
    test_results.to_csv(f'{result_project_folder}/{project}_test_result_cross.csv', index=False, float_format='%.3f')
    result_df.to_csv(f'{result_project_folder}/{project}_result_cross.csv', index=False, float_format='%.3f')

    # process and dump feature importance
    feature_importance_df = pd.DataFrame({'feature': feature_list, 'importance': np.mean(feature_importances, axis=0)})
    feature_importance_df['importance'] = feature_importance_df['importance'] * 100 / feature_importance_df[
        'importance'].sum()
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(f'{result_project_folder}/{project}_feature_importance_cross.csv', index=False,
                                 float_format='%.3f')
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
            if run: total_result += result_df
            else: total_result = result_df

        total_result /= runs
        print(total_result.mean())
        print()


if __name__ == '__main__':
    main()