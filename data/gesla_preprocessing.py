#- Modules
import numpy as np
import xarray as xr
import pandas as pd

#- Main

def stations_as_coordinates(ds):
    """
    This function replaces the 'station' dimension in an xarray Dataset
    with coordinates based on a provided station dictionary.

    Parameters:
    ds (xarray.Dataset): The input dataset with a 'station' dimension.
    station_dict (dict): A dictionary where keys are station indices and values are the station names.

    Returns:
    xarray.Dataset: Dataset with 'station' replaced by the corresponding names from station_dict.
    """
    station_dict = {}
    for station_id in ds.station.values:
        station_name = ds.sel(station=station_id).file_name.dropna("date_time").values[0]
        station_dict[station_id] = station_name
        
    # Convert station_dict to DataArray for easier manipulation
    station_names = xr.DataArray(list(station_dict.values()), dims='station')
    
    # Assign the new station names as coordinates
    ds = ds.assign_coords(station=station_names)
    
    return ds
    
def is_winter(month):
    """
    Description:
        Mask for winter season, e.g. December, January and February.

    Parameters:
        month (xr.DataArray): Containing the month of a timeseries
    
    Returns:
        Boolean mask for winter season
    """

    return (month == 1) | (month == 2) | (month == 12)

def is_autumn(month):
    """
    Description:
        Mask for autumn season, e.g. Sep, Oct and Nov.

    Parameters:
        month (xr.DataArray): Containing the month of a timeseries
    
    Returns:
        Boolean mask for autumn season
    """

    return (month == 9) | (month == 10) | (month == 11)

def get_analysis(ds):
    """
    Description: 
        Selects all values of GESLA data where use_flag == 1.
        Drops all NaN values.

    Parameters:
        ds (xr.Dataset): GESLA Dataset for several stations

    Returns:
        df (pd.Dataframe): 
    """
    ds = ds.where(ds.use_flag == 1., drop = True) # Analysis flag
    df = ds.to_dataframe().dropna(how="all") 

    return df
    
def get_station_names(gesla: xr.Dataset()) -> dict:
    """
    Description: Returns the sorted station names of the GESLA dataset as np.array.
    Parameters:
        gesla: xr.Dataset(): GESLA-Dataset with station names saved in "filename".
    Returns:
        A dictionary with the station names and their index as sorted in given GESLA dataset.
    """
    fnames = gesla["filename"].values.astype(str)
    station_names = {}
    for i, station_name in enumerate(fnames): # Loop over all stations
        # name_idx = np.where(np.unique(fnames[i,:]) != "nan") ?? Use this if analysis flag of gesla was chosen before applying this function.
        # station_name = np.unique(fnames[i])[name_idx][0] # Select station name from array
        # station_names[station_name] = i
        station_names[station_name] = i
    return station_names
    
def station_position(gesla_predictand, station_names):
    """
    Description:
        Returns positions longitudes and latitudes of GESLA Stations
    Parameters:
        gesla_predictand (xr.Dataset): GESLA Dataset loaded with data_loader.load_gesla(station_names)
        station_names (dict): dictionary of station names and their index within GESLA dataset
    Returns:
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
    """
    lons = gesla_predictand.longitude.values
    lats = gesla_predictand.latitude.values

    station_positions = {}
    for station_name, station_idx in station_names.items():
        lon = filter_lon_lat(lons, station_idx)
        lat = filter_lon_lat(lats, station_idx)
        station_positions[station_name] = [lon, lat]

    return station_positions  
    
def detrend(df, level="station"):
    """
    Description:
        Detrends pd.Series by subtracting mean from specified index / level.
        Data is grouped by level.

    Parameters:
        df (pd.Series): Dataframe with timeseries data
        level (str): Index along which to subtract mean (Default:"station")

    Returns:
        pd.Series: Detrended dataframe for each index of level
    """
    return (df - df.groupby(level=level).mean())

def detrend_signal(gesla_predictand, type_):
    """
    Description:
        Detrend GESLA-Dataset either by subtracting mean or linear trend
    Parameters:
        gesla_predictand (pd.Series): Gesla dataset at selected station. 
        type_ (str): "constant" or "linear" for either subtracting mean or linear trend, respectively.
    Returns:
        df (pd.Series): Detrended Gesla Dataseries.
    Note: 
        Only one station can be selected. This is not grouped by stations.
        Returns in the format as expected by further modules.
    """
    from scipy import signal
    import pandas as pd
    import numpy as np

    detrended = signal.detrend(gesla_predictand, type=type_,)

    # Get date_time for index
    #---
    date_time = []
    for index in gesla_predictand.index.values:
        date = index[1]
        date_time.append(date)

    # Create new dataframe
    #---
    station_idx = np.zeros(detrended.shape).astype(int)
    df = pd.DataFrame(
        {"sea_level": detrended},
        index=[
            station_idx,
            date_time,
        ],
        
    )
    df.index.names = ["station", "date_time"]
    
    return df.squeeze()

def apply_dummies(df, percentile=0.95, level="station"):
    """
    Description:
        Applies one-hot encoding on dataseries for specified percentile along
        an index. Labels data with 1, if datapoint is in percentile.

    Parameters:
        df (pd.Series): Dataseries with timeseries data
        percentile (float): Percentile to evaluate dummies (Default: 0.95)
        level (str): Index along which to subtract mean (Default: "station")

    Returns:
        dummies (pd.Series): DataFrame with labeled data (1 if data is in percentile, 0 if not.)
    """
    dummies = df - df.groupby(level=level).quantile(percentile)
    dummies[dummies >= 0] = 1
    dummies[dummies < 0] = 0

    return dummies

def select_season(ds_gesla, season):
    """
    Description:
        Selects a season of GESLA dataset
    Parameters:
        ds_gesla (xr.Dataset): GESLA dataset 
        season (str): "winter" or "autumn"
    Returns:
        season_ds (xr.Dataset): GESLA Dataset for specific month of the season.
    """
    # Modules
    #---
    from data import gesla_preprocessing

    if season == "autumn":
        get_season = gesla_preprocessing.is_autumn
    elif season == "winter":
        get_season = gesla_preprocessing.is_winter
    else:
        raise Exception(f"season: {season} is not available in this process")

    season_ds = ds_gesla.sel(date_time=get_season(ds_gesla['date_time.month']))

    return season_ds
    
def filter_lon_lat(data:np.array, 
                   station_idx:int,):
    """Filters a single longitude or latitude of a station.
    It filters NaN-values and return the single lon or lat for the given station.
    Parameters:
        data (np.array) : The np.array containing lon or lat values of the gesla data set
        station_idx (int) : the index of the station to select the data from.
    """
    _data = np.unique(data[station_idx])
    nan_filter = ~np.isnan(_data)
    data = _data[nan_filter][0]
    return data
# def station_position(gesla_predictand, station_names):
#     """
#     Description:
#         Returns positions longitudes and latitudes of GESLA Stations
#     Parameters:
#         gesla_predictand (xr.Dataset): GESLA Dataset loaded with data_loader.load_gesla(station_names)
#         station_names (list): List of station names within GESLA dataset
#     Returns:
#         station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
#     """
#     lon = gesla_predictand.longitude.dropna("date_time").values
#     lat = gesla_predictand.latitude.dropna("date_time").values

#     station_positions = {}
#     for station_idx, station_name in enumerate(station_names):
#         lon = gesla_predictand.isel(station=station_idx).longitude.dropna("date_time").values[0]
#         lat = gesla_predictand.isel(station=station_idx).latitude.dropna("date_time").values[0]
#         station_positions[station_name] = [lon, lat]

#     return station_positions