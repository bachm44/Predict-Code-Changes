from sklearn.ensemble import RandomForestClassifier
import time
from ..Util import *


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

    # # hyper-tuning
    # for n_estimators in [100, 500]:
    #     for max_depth in [None, 2, 3, 5, 10]:
    #         print(n_estimators, max_depth)
    #         run_model(RandomForestClassifier(class_weight='balanced', n_estimators=n_estimators,max_depth=max_depth),
    #                   df, feature_list=feature_list)
    best_n_estimators = 500
    best_max_depth = 5

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
            clf = RandomForestClassifier(class_weight='balanced', n_estimators=best_n_estimators,
                                         max_depth=best_max_depth)
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

    train_results.to_csv(f'{result_project_folder}/{project}_train_result_fan_cross.csv', index=False, float_format='%.3f')
    test_results.to_csv(f'{result_project_folder}/{project}_test_result_fan_cross.csv', index=False, float_format='%.3f')
    result_df.to_csv(f'{result_project_folder}/{project}_result_fan_cross.csv', index=False, float_format='%.3f')
    print()


if __name__ == '__main__':
    main()