#---
# Description:
#   Functions that give information on dataframes like duplicated indices, overlap of timeintervals etc.
#---

#- Modules
import numpy as np

#- Main
def get_duplicated_idx(df):
    """
    Description: 
        Checks for duplicate indices in dataframe

    Parameters:
        df(pd.Dataframe)

    Output: 
        Index position of duplicates
    """

    if np.where(df.index.duplicated() == True)[0].size == 0:
        return None
    else:
        return np.where(df.index.duplicated() == True)[0]

def assert_equality(t1, t2):
    """
    Description: 
        Checks if two arrays are the same. Throws assertion error if not.
    Parameters:
        t1, t2 (np.array): Arrays of data points, shape:(datapoints,)
    """

    if len(np.where(t1 != t2)[0]) == 0:
        is_equal = True
    else:
        is_equal = False

    assert is_equal, "Arrays are not equal"
    print("Arrays are the same")