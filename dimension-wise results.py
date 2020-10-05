from utility import *
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

feature_df = load_features(project)
feature_list = get_initial_feature_list()

# scaling
scaler = StandardScaler()
feature_df[feature_list] = scaler.fit_transform(feature_df[feature_list])

temps = {}
for group in features_group:
    print(group)
    feature_sub_list = features_group[group]
    result = Result()
    y_probs = []
    for fold in range(1, folds):
        train_size = feature_df.shape[0]*fold // folds
        test_size = min(feature_df.shape[0] * (fold + 1)//folds, feature_df.shape[0])

        x_train, y_train = feature_df.loc[:train_size-1, feature_sub_list], feature_df.loc[:train_size-1, target]
        x_test, y_test = feature_df.loc[train_size:test_size-1, feature_sub_list], feature_df.loc[train_size:test_size-1, target]

        clf = LGBMClassifier(objective='binary', class_weight='balanced', random_state=seed)
        clf.fit(x_train, y_train)

        y_prob = clf.predict_proba(x_test)[:, 1]
        result.calculate_result(y_test, y_prob, fold, False)
        y_probs.extend(y_prob)

    result_df = result.get_df()
    print(result_df.mean())
    print()
