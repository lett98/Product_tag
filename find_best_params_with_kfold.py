import fasttext
import copy
import tempfile
import shutil
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def get_gridsearch_params(param_grid):
    params_combination = [dict()]  # Used to store all possible parameter combinations
    for k, v_list in param_grid.items():
        tmp = [{k: v} for v in v_list]
        n = len(params_combination)
        copy_params = [copy.deepcopy(params_combination) for _ in range(len(tmp))] 
        params_combination = sum(copy_params, [])
        _ = [params_combination[i*n+k].update(tmp[i]) for k in range(n) for i in range(len(tmp))]
    return params_combination

#Calculate classification evaluation index
def get_metrics(y_true, y_pred):
    metrics = {}

    average = 'macro'
    # metrics[average+'_precision'] = precision_score(y_true, y_pred, average=average)
    # metrics[average+'_recall'] = recall_score(y_true, y_pred, average=average)
    metrics[average+'_f1'] = f1_score(y_true, y_pred, average=average)  
    return metrics
 
# Use k-fold cross validation to get the final score and save the best score and its corresponding set of parameters
# The inputs are the training data frame, the parameters to be searched, the KFold object for cross validation, the best score evaluation index, and several classifications
def get_KFold_scores(df, params, kf, metric):
    metric_score = 0.0

    for train_idx, val_idx in kf.split(df['product_name'],df['category']):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        tmpdir = tempfile.mkdtemp() #Because the directory or file where the training data is located is read during the training of fasttext, a temporary directory / file is opened with the cross validation set
        tmp_train_file = tmpdir + '/train.txt'
        df_train.to_csv(tmp_train_file, sep='\t', index=False, header=None, encoding='UTF-8')  # No header
        
        
        fast_model = fasttext.train_supervised(tmp_train_file, **params) #Training, incoming parameters
        
        #Use the trained model for evaluation and prediction
        predicted = fast_model.predict(df_val['product_name'].tolist())  # ([label...], [probs...])
        y_val_pred = predicted[0]  # label[0]  __label__0
        y_val = df_val['category'].values

        
        score = get_metrics(y_val, y_val_pred)[metric]
        print('score:' ,score)
        metric_score += score #The scores accumulated on different training sets are used to calculate the average score on the whole cross validation set
        shutil.rmtree(tmpdir, ignore_errors=True) #Delete temporary training data file

    print('average macro_f1:', metric_score / kf.n_splits)
    return metric_score / kf.n_splits

# Grid search + cross validation
# The input is the training data frame, the parameters to be searched, the best score evaluation index, and how much discount should be made for cross validation
def my_gridsearch_cv(df, param_grid, metrics, kfold=5):
    n_classes = len(np.unique(df['category']))
    print('n_classes', n_classes)

    skf = StratifiedKFold(n_splits=kfold,shuffle=True,random_state=1) #k-fold stratified sampling cross validation

    params_combination = get_gridsearch_params(param_grid) # Get various permutations and combinations of parameters

    best_score = 0.0
    best_params = dict()
    for params in params_combination:
        avg_score = get_KFold_scores(df, params, skf, metrics)
        if avg_score > best_score:
            best_score = avg_score
            best_params = copy.deepcopy(params)

    return best_score, best_params

# Parameters to debug
tuned_parameters = {
    'lr': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0],
    'epoch': [50],
    'wordNgrams': [2],
}

if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    best_score, best_params = my_gridsearch_cv(train, tuned_parameters, 'macro_f1', kfold=5)
    print('best_score', best_score)
    print('best_params', best_params)
