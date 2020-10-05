from utility import *
from config import *
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

feature_df = load_features(project)
feature_list = get_initial_feature_list()

# scaling
scaler = StandardScaler()
feature_df[feature_list] = scaler.fit_transform(feature_df[feature_list])

scores = [0] * 10
for fold in range(1, folds):
    train_size = feature_df.shape[0]*fold // folds
    test_size = min(feature_df.shape[0] * (fold + 1)//folds, feature_df.shape[0])

    x_train, y_train = feature_df.loc[:train_size-1, feature_list], feature_df.loc[:train_size-1, target]
    x_test, y_test = feature_df.loc[train_size:test_size-1, feature_list], feature_df.loc[train_size:test_size-1, target]

    clf = LGBMClassifier(objective='binary', class_weight='balanced', random_state=seed)
    clf.fit(x_train, y_train)

    y_prob = clf.predict_proba(x_test)[:, 1]

    for k in range(10, 100, 10):
        score = Result.cost_effectiveness(y_test, y_prob, k)
        scores[k//10] += score

# averaging over folds
for k in range(10, 100, 10):
    print(k, scores[k//10]/(folds-1))
