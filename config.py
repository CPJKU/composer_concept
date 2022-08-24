import os

# please adapt accordingly
maestro_root = '/path/to/maestro-v2.0.0'
asap_root = '/path/to/asap-dataset'

# no need to change these (unless you want to)
project_root = '.'
data_root = os.path.join(project_root, 'preprocessed_data')
results_root = os.path.join(project_root, 'results')
splits_root = os.path.join(project_root, 'classifier', 'meta')
concepts_path = os.path.join(project_root, 'concepts')