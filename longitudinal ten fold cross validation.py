from utility import *
from config import *
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import time

feature_df = load_features(project)
feature_list = get_initial_feature_list()

# scaling
scaler = StandardScaler()
feature_df[feature_list] = scaler.fit_transform(feature_df[feature_list])

runs = 2
train_results = None
test_results = None
feature_importance = []
train_time = {fold: [] for fold in range(1, folds)}
test_time = {fold: [] for fold in range(1, folds)}
feature_list = get_initial_feature_list()

for run in range(runs):
    print(f"Running {run+1} th iteration")
    test_result = Result()
    train_result = Result()

    for fold in range(1, folds):
        train_size = feature_df.shape[0] * fold // folds
        test_size = min(feature_df.shape[0] * (fold + 1) // folds, feature_df.shape[0])

        x_train, y_train = feature_df.loc[:train_size - 1, feature_list], feature_df.loc[:train_size - 1, target]
        x_test, y_test = feature_df.loc[train_size:test_size - 1, feature_list], \
                         feature_df.loc[train_size:test_size - 1, target]

        start = time.time()
        clf = LGBMClassifier(class_weight='balanced', random_state=seed)
        clf.fit(x_train, y_train)
        train_time[fold].append(time.time() - start)

        y_prob = clf.predict_proba(x_train)[:, 1]
        train_result.calculate_result(y_train, y_prob, fold, False)

        start = time.time()
        y_prob = clf.predict_proba(x_test)[:, 1]
        test_time[fold].append(time.time() - start)
        test_result.calculate_result(y_test, y_prob, fold, False)
        feature_importance.append(clf.feature_importances_)

    train_result_df = train_result.get_df()
    test_result_df = test_result.get_df()

    if run:
        train_results += train_result_df
        test_results += test_result_df
    else:
        train_results = train_result_df
        test_results = test_result_df

train_results /= runs
test_results /= runs

result_df = pd.DataFrame({"train": train_results.mean(), "test": test_results.mean()}).reset_index()
print(result_df)

# average time across runs
for fold in range(1, folds):
    train_time[fold] = np.mean(train_time[fold])
    test_time[fold] = np.mean(test_time[fold])
train_results['time'] = train_time.values()
test_results['time'] = test_time.values()


# process feature importance
feature_importance_df = pd.DataFrame({'feature': feature_list, 'importance': np.mean(feature_importance, axis=0)})
feature_importance_df['importance'] = feature_importance_df['importance'] * 100 / feature_importance_df[
    'importance'].sum()
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
print(feature_importance_df)

# dump data in result directory
output_dir = f'{root}/result'
train_results.round(3).to_csv(f'{output_dir}/{project}_train_results.csv', index=False)
test_results.round(3).to_csv(f'{output_dir}/{project}_test_results.csv', index=False)
feature_importance_df.round(3).to_csv(f'{output_dir}/{project}_feature_importance.csv', index=False)
result_df.round(3).to_csv(f'{output_dir}/{project}_result.csv', index=False)

print()
