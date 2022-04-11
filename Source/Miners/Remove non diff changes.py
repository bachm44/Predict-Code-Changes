import joblib
import os

import pandas as pd

from Config import diff_root, change_list_filepath, selected_change_list_filepath, selected_changes_path


def remove_changes_without_diff():
    change_numbers = [int(filename.split('_')[1]) for filename in os.listdir(diff_root)]
    selected_changes = pd.read_csv(selected_changes_path)
    old_number = selected_changes.shape[0]
    selected_changes = selected_changes[selected_changes['change_id'].isin(change_numbers)]

    print("Changes reduced from {0} to {1} after removing those without diff file.".format(
        old_number, selected_changes.shape[0]))
    selected_changes.to_csv(selected_changes_path, index=False)

    df = pd.read_csv(change_list_filepath)
    df = df[df['change_id'].isin(selected_changes['change_id'].values)]
    df.to_csv(selected_change_list_filepath, index=False)


if __name__ == '__main__':
    # remove changes from the selected changes list for which file diff content was not found
    remove_changes_without_diff()
