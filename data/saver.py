def save_hp(hparams, hparam_names, predictor, model_run, percentile, folder, is_overwrite=False):
    """
    Description: 
        Saves Hyperparameters manually read from validation curves in 
        master_thesis/results/random_forest/rf002.
        Saves to "models/random_forest/hyperparameter/" 

    Parameters:
        hparams (tuple): Fixed values of hyperparameters
        hparam_names (tuple): Names of hyperparameters in hparams
        predictor (str): Flag of predictor ["sp", "tp", "u10", "v10"]
        model_run (str): Flag for model run.
        percentile (float): percentile of preprocessing predictand.
        folder (str): Folder where data is saved.
        is_overwrite (bool): Whether to overwrite an existing file or create a new file. (Defaults: False)
    
    Returns:
        None
    """
    # Modules
    import pickle

    # Initialize
    params = dict(zip(hparam_names, hparams))
    fname = f"{model_run}_{predictor}{str(percentile)[-2:]}.pkl"
    file_path = f"{folder}{fname}"

    # Check if folder exists, otherwise create it
    directory_existance(folder)

    # If files should not be overwritten, give unique name
    if not is_overwrite:
        file_path = unique_filenames(file_path)

    # Save file
    with open(f"{file_path}", 'wb') as f:
        pickle.dump(params, f)
        print(f"File saved to: {file_path}")


def save_hpdict(hparams, predictor, model_run, percentile, folder, is_overwrite=False):
    """
    Description: 
        Saves Hyperparameters manually read from validation curves in 
        master_thesis/results/random_forest/rf002.
        Saves to "models/random_forest/hyperparameter/" 

    Parameters:
        hparams (dict): dictionary of hyperparameters
        predictor (str): Flag of predictor ["sp", "tp", "u10", "v10"]
        model_run (str): Flag for model run.
        percentile (float): percentile of preprocessing predictand.
        folder (str): Folder where data is saved.
        is_overwrite (bool): Whether to overwrite an existing file or create a new file. (Default:False)

    
    Returns:
        None
    """
    # Modules
    import pickle

    # Initialize
    fname = f"{model_run}_{predictor}{str(percentile)[-2:]}.pkl"
    file_path = f"{folder}{fname}"

    # Check if folder exists, otherwise create it
    directory_existance(folder)

    # If files should not be overwritten, give unique name
    if not is_overwrite:
        file_path = unique_filenames(file_path)
    
    # Save file
    with open(f"{file_path}", 'wb') as f:
        pickle.dump(hparams, f)
        print(f"File saved to: {file_path}")

#---
# Auxillary Functions
#---

def directory_existance(directory):
    """
    Description:
        Checks whether a directory exists and creates a folder if it is non-existent.
    Parameters:
        directory (str): Path to a folder
    Returns:
        None
    """

    # Modules
    import os

    # Check if directory exists
    is_exist = os.path.exists(directory)

    # If directory does not exist, create it
    if not is_exist:
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def unique_filenames(file_name):
    """
    Description:
        Checks if a file already exists. If so, it gives a new name to the file by adding a number to the end.
    Parameters:
        file_name (str): Whole path to the file, e.g. notebooks/create_gifs.ipynb
    Returns:
        file_name (str): Adjusted path to the file, if file already existed, e.g. notebook/create_gifs1.ipynb

    """
    import os

    if os.path.isfile(file_name):
        expand = 0
        while True:
            expand += 1
            fname = file_name.split(".")[0]
            file_format = file_name.split(".")[1]
            new_file_name = f"{fname}{str(expand)}.{file_format}"

            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break

    return file_name
        
