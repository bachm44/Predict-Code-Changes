from Config import *
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from Source.Util import Result

best_n_estimators = 500
best_learning_rate = 0.01


def get_best_model():
    return LGBMClassifier(class_weight='balanced', n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


scaler = StandardScaler()
feature_list = initial_feature_list
target = 'status'

data = {
    project: pd.read_csv(f'{data_folder}/{project}/{project}.csv', encoding='utf-8') for project in projects
}

for current_project in projects:
    print(current_project)
    df = data[current_project]

    df[feature_list] = scaler.fit_transform(df[feature_list], df[target])

    for other_project in projects:
        if current_project == other_project: continue

        other_df = data[other_project]
        other_df[feature_list] = scaler.transform(other_df[feature_list])
        y_true = other_df[target]

        results = []
        for run in range(runs):
            model = get_best_model()
            model.fit(df[feature_list], df[target])

            y_prob = model.predict_proba(other_df[feature_list])[:, 1]
            auc = roc_auc_score(y_true, y_prob)
            cost_effectiveness = Result.cost_effectiveness(y_true, y_prob, 20)

            y_pred = np.round(y_prob)
            f1_m, f1_a = f1_score(y_true, y_pred), f1_score(y_true, y_pred, pos_label=0)
            results.append([auc, cost_effectiveness, f1_m, f1_a])

        results = np.mean(results, axis=0)
        print(f'{other_project}: {np.round(results, 3)}')
        print()


# Libreoffice
# Eclipse: [0.643 0.951 0.852 0.23 ]
#
# Gerrithub: [0.669 0.925 0.887 0.322]
#
# Eclipse
# Libreoffice: [0.775 0.975 0.846 0.306]
#
# Gerrithub: [0.766 0.978 0.88  0.36 ]
#
# Gerrithub
# Libreoffice: [0.792 0.983 0.844 0.307]
#
# Eclipse: [0.811 0.959 0.876 0.502]