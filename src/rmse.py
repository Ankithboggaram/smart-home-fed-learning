import numpy as np

def rmse(y_true, y_pred):
    '''
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    '''
    # Compute RMSE
    return np.sqrt(np.mean((y_true - y_pred) ** 2))