#- Modules
import numpy as np
import xarray as xr
import pandas as pd
from data import gesla_preprocessing

#- Main
def load_hourly_era5(range_of_years, subregion, season, predictor, era5_import):
    """
    Description:
        Loads hourly ERA5 Data for a specified range of years, 
        subregion and season for one predictor.

    Parameters:
        range_of_years (str): range of years, "year1-year2"
        subregion (str): Lon / Lat subregion of 'lon-7050_lat8030'
        season (str): winter or autumn
        predictor (str): Predictor of Storm Surges
        era5_import (str): Folder of preprocessed data

    Returns:
        da (xr.DataArray): Hourly xarray-DataArray for the predictor

    Note:
        Parameter-Section needs to be updated, if new data was preprocessed.
    """

    # Parameter
    available_years = [
    "1999-2008",
    "2009-2018",
    "2019-2022",
    ]

    available_predictors = [
        "sp",
        "tp",
        "u10",
        "v10",
    ]

    available_subregions = [
        "lon-0530_lat7040", # preprocess1
    ]

    available_seasons = [
        "winter",
        "autumn",
    ]

    available_era5_imports = [
        "preprocess1",
    ]

    # Error Handling
    assert range_of_years in available_years, f'years: {range_of_years} not in available set of {available_years}'
    assert subregion in available_subregions, f'subregion: {subregion} not in available set of {available_subregions}'
    assert season in available_seasons, f'season: {season} not in available set of {available_seasons}'
    assert predictor in available_predictors, f'predictor: {predictor} not in available set of {available_predictors}'
    assert era5_import in available_era5_imports, f'era5_import: {era5_import} not in available set of {available_era5_imports}'

    # Main
    folder = f"resources\era5\{era5_import}\\"

    file = f'{folder}{predictor}_{range_of_years}_{subregion}_{season}.nc'

    ds = xr.open_dataset(file)

    da = ds[predictor]

    return da

def load_daymean_era5(range_of_years, subregion, season, predictor, era5_import):
    """
    Description:
        Loads daily mean of ERA5 Data for a specified range of years, 
        subregion and season for one predictor.

    Parameters:
        range_of_years (str): range of years, "year1-year2"
        subregion (str): Lon / Lat subregion of 'lon-7050_lat8030'
        season (str): winter or autumn
        predictor (str): Predictor of Storm Surges
        era5_import (str): era5_importing Folder

    Returns:
        ds (xr.DataArray): Hourly xarray-DataArray for the predictor

    Note:
        Parameter-Section needs to be updated, if new data was preprocessed.
    """
    #---
    # Parameter
    #---
    available_years = [
    "1999-2008",
    "2009-2018",
    "2019-2022",
    ]

    available_predictors = [
        "sp",
        "tp",
        "u10",
        "v10",
    ]

    available_subregions = [
        "lon-0530_lat7040", # preprocess1
    ]

    available_seasons = [
        "wdmean", # winter
        "admean", # autumn
    ]

    available_era5_imports = [
        "preprocess1",
    ]

    #---
    #  Preprocessed Parameters
    #---
    if season.lower() == "winter":
        season = "wdmean"
    if season.lower() == "autumn":
        season = "admean"

    #---
    #  Error Handling
    #---    
    assert range_of_years in available_years, f'years: {range_of_years} not in available set of {available_years}'
    assert subregion in available_subregions, f'subregion: {subregion} not in available set of {available_subregions}'
    assert season in available_seasons, f'season: {season} not in available set of {available_seasons}'
    assert predictor in available_predictors, f'predictor: {predictor} not in available set of {available_predictors}'
    assert era5_import in available_era5_imports, f'Preprocess: {era5_import} not in available set of {available_era5_imports}'

    #---
    #  Main
    #---
    folder = f"resources\era5\{era5_import}\\"

    file = f'{folder}{predictor}_{range_of_years}_{subregion}_{season}.nc'

    ds = xr.open_dataset(file)

    da = ds[predictor]
    
    return da

def load_gesla(station_names, meta_file, data_path):
    """
    Description: 
        Loads GESLA Dataset at indicated stations
    Parameters:
        station_names (list): A list of station flags of the GESLA Dataset. 
        meta_file (string): Path to the GESLA.csv Meta File
        data_path (string): Path to the GESLA.zip file
    Returns:
        ds (xr.Dataset): Sea Level values at indicated stations
    """
    # Modules
    #---
    from gesla import GeslaDataset

    print("Load Predictand from GESLA")

    # Create GESLA Dataset
    g3 = GeslaDataset(meta_file=meta_file, data_path=data_path)

    # Select Stations
    ds = g3.files_to_xarray(station_names)

    return ds

#---
# Load data of predictor
#---
def load_pf(season):
    """
    Description:
        Loads sea level values at Degerby Station and uses them as a proxy for prefilling of the Baltic Sea.
    Parameters:
        season (str): "winter" or "autumn"
    Returns:
        degerby_proxy (xr.DataArray): Preprocessed sea level values of sea level at Degerby Station.
    """
    station_names = ['degerby-deg-fin-cmems',]
    degerby_proxy = load_gesla(station_names)

    # Select a season
    #---
    degerby_proxy = gesla_preprocessing.select_season(degerby_proxy, season)

    # Select only sea_level analysis data
    #---
    degerby_proxy = gesla_preprocessing.get_analysis(degerby_proxy)
    degerby_proxy = degerby_proxy["sea_level"] # Select values
    degerby_proxy = degerby_proxy.to_xarray()

    return degerby_proxy