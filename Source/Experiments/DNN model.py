from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from ..Util import *


def build_model(n):
    # Creating the model
    model = Sequential()

    # Inputing the first layer with input dimensions
    model.add(Dense(n, activation='relu', input_dim=n))
    model.add(Dropout(0.10))

    model.add(Dense(16, activation='relu'))

    # Adding another Dropout layer
    model.add(Dropout(0.10))

    # adding the output layer that is binary [0,1]
    model.add(Dense(1, activation='sigmoid'))
    # With such a scalar sigmoid output on a binary classification problem, the loss
    # function you should use is binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC()])

    # Visualizing the model
    # model.summary()
    return model


feature_list = initial_feature_list
print(project)
df = pd.read_csv(f'{root}/{project}.csv')
df[feature_list] = StandardScaler().fit_transform(df[feature_list])

results = None
for run in range(runs):
    result = Result()
    for fold in range(1, folds):
        train_size = df.shape[0] * fold // folds
        test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

        train, test = df.iloc[:train_size], df.iloc[train_size:test_size]
        train_x, train_y = train[feature_list], train[target]
        test_x, test_y = test[feature_list], test[target]
        model = build_model(len(feature_list))

        # weighted loss
        weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=train_y)
        model.fit(train_x, train_y, epochs=10, verbose=10, class_weight={0: weights[0], 1: weights[1]})

        y_prob = model.predict(test_x).flatten()
        result.calculate_result(test_y, y_prob, fold, False)

    result_df = result.get_df()
    if run:
        results += result_df
    else:
        results = result_df

results /= runs
print(results.mean())
print()
