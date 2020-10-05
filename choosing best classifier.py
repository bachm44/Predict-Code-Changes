from utility import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

feature_df = load_features(project)
feature_list = get_initial_feature_list()

# scaling
scaler = StandardScaler()
feature_df[feature_list] = scaler.fit_transform(feature_df[feature_list])

models = [
    GradientBoostingClassifier(n_estimators=100, random_state=seed),
    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed),
    ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=seed),
    LogisticRegression(max_iter=100, class_weight='balanced', random_state=seed),
    LGBMClassifier(class_weight='balanced', random_state=seed)
]

for model in models:
    print(f'Currently running {model.__class__.__name__}')
    test_result = Result()
    for fold in range(1, folds):
        train_size = feature_df.shape[0] * fold // folds
        test_size = min(feature_df.shape[0] * (fold + 1) // folds, feature_df.shape[0])

        x_train, y_train = feature_df.loc[:train_size - 1, feature_list], feature_df.loc[:train_size - 1, target]
        x_test, y_test = feature_df.loc[train_size:test_size - 1, feature_list], feature_df.loc[
                                                                                 train_size:test_size - 1, target]

        clf = model
        clf.fit(x_train, y_train)

        y_score = clf.predict_proba(x_test)[:, 1]
        test_result.calculate_result(y_test, y_score, fold, False)

    print(test_result.get_df().mean())
    print()