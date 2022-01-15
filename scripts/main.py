import argparse
import os
import torch

from data_ops import read_compress, read_dataset
from model_utils import train_model, import_model


def configure_args():
    """ Return CLI arguments. """
    
    parser = argparse.ArgumentParser(description = 'Argument name space for CLI flags.')
    
    parser.add_argument('--data_dir', type = str, help = 'Data directory',
                        default = os.path.join(os.getcwd().replace('scripts', 'data'),
                                               'ct-dataset.csv'))
    
    parser.add_argument('--arch_dir', type = str, default = os.path.join(os.getcwd().replace('scripts', 'data'),
                                                                         'ct-dataset.zip'))
    
    parser.add_argument('--style', type = str, default = 'gruvboxd', help = 'Visualization style',
                        choices = ['gruvboxd', 'solarizedd', 'solarizedl', 'normal', 'chesterish'])
    
    parser.add_argument('--epochs', type = int, default = 20, help = 'Number of complete data rounds')
    
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Convergence rate')
    
    parser.add_argument('--classes', default = None, help = 'Number of classes')
    
    parser.add_argument('--n_features', type = int, default = 348, help = 'Data dimentionality')
    
    parser.add_argument('--cardinality', type = int, default = 50, help = 'Max number of possible classes')
    
    parser.add_argument('--var_lim', type = float, default = 0.8, help = 'Max allowable limit on variance')
    
    parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size')
    
    parser.add_argument('--t1', type = float, default = 0.2, help = 'Train split')
    
    parser.add_argument('--t2', type = float, default = 0.35, help = 'Valid split')
    
    parser.add_argument('--split', type = float, default = 0.2, help = 'Validation split while fitting?')
    
    parser.add_argumet('--valid', type = bool, choice = [True, False], default = True, help = 'Validation split?')
    
    parser.add_arhument('--show_valid', type = bool, default = True, choices = [True, False],
                        help = 'Show validation metrrics?')
    
    parser.add_argument('--save', type = bool, default = True, choices = [True, False], help = 'Save model artefacts?')
    
    parser.add_argument('--artefact_dir', type = bool, default = os.getcwd().replace('scripts', 'artefacts'),
                        help = 'Location for saved model')
    
    parser.add_argument('--task', default = 'classif', type = str, choices = ['classif', 'regression'],
                        help = 'Type of experience, E')
    
    return parser



def main():
    !pip install jupyterthemes
    print('>>> Configure CLI arguments...')
    args = configure_args().parse_args()
    print('>> CLI arguments configured!')
    print()
    
    print('>>> Importing dataset...')
    data, path_to_archive = read_compress(path_to_data = args.data_dir,
                                          path_to_archive = args.arch_dir)
    print('>>> Dataset imported successfully!')
    print()
    
    print('>>> Data preprocessing underway...')
    drop_labels = get_invariant_features(X, cardinality = args.cardinality, percent = args.var_lim)
    
    X, y = variables(data, labels = drop_labels, target = 'reference', return_y = True)
    
    X, y = array_to_tensor(X), array_to_tensor(y)
    
    print('>>> Data preprocessing complete!')
    print()
    
    ### Split dataset for training, validation, and testing
    print('>>> Partitioning data into splits...')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size = args.t1)
    
    if args.valid:
        X_train, X_val, y_train, y_val = split_data(X_train, y_train, test_size = args.t2)
        
    print('>>> Data splits created!')
    print()
    
    ### Instantiate model object
    model = import_model(task = args.task, n_features = args.n_features, n_classes = args.classes,
                         optim = torch.optim.SGD, lr = args.lr, max_epochs = args.max_epochs,
                         batch_size = args.batch_size, split = args.split)
    
    ### Train model
    print('>>> Model training underway...')
    model = train_model(X_train, y_train)
    
    print('>>> Model training complete!')
    print()
    
    print('>>> Displaying diagnostics...')
    print()
    
    print(get_r2_score(y_train, model.predict(X_train), num_places=5, text=True))
    print()
    
    if args.show_valid:
        print(get_r2_score(y_valid, model.predict(X_valid), num_places=5, text=True))
        print()
        
    print(get_r2_score(y_test, model.predict(X_test), num_places=5, text=True))
    
    if args.save:
        os.makedirs(args.artefact_dir)
        DIR = os.path.join(args.artefact_dir, 'model.pkl')
        
        with open(DIR, 'wb') as f:
            pickle.dump(model, f)













                        
                        
