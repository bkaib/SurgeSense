def load_model(model_path):
    """
    Description: 
        Loads a saved model (.sav) or hyperparameters (.pkl) from disk.
    Parameters:
        model_path (str): Path to modelfile (.pkl or .sav)
    Returns:  
        model (dict): classifier with fitted hyperparameters of the model run.
    """
    import pickle

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model

def load_scaler(path: str):
    """
    Description:
        Loads the scaler of a model from path.
    Parameters:
    Returns:
    """
    import pickle

    with open(path,'rb') as f:
        scaler = pickle.load(f)

    return scaler