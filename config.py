# Can be 'Eclipse', 'Openstack'
project = 'Libreoffice'
seed = 7
folds = 11
root = 'data'
target = 'status'

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num'],
    'text': ['description_length','is_documentation','is_bug_fixing','is_feature'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author'],
    'reviewer': ['reviewers_num', 'avg_reviewer_experience', 'avg_reviewer_review_count'],
    'code': ['lines_added','lines_deleted','files_added','files_deleted', 'files_modified',
             'directory_num', 'modify_entropy']
}
