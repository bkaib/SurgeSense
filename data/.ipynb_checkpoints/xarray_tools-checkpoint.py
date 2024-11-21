import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt


def get_N_closest(
        lons : xr.DataArray, 
        lats : xr.DataArray, 
        target_lon : float, 
        target_lat : float, 
        N : int = 1,
) -> np.ndarray:
    """
    Description:
        Computes the N closest neighbours to a given lon/lat position based on Least Squares
    Parameters:
        lons : longitudes of the DataArray
        lats : latitudes of the DataArray
        target_lon: longitude of interest
        target_lat: latitude of interest
        N : Number of nearest neighbours
    Returns:
        indices: indices within the DataArray of N closest neighbours
    """
    # Calculate Distances
    distances = np.sqrt((lons - target_lon)**2 + (lats - target_lat)**2)

    # Sort distances and get indices of N smallest ones
    indices = np.unravel_index(np.argsort(distances.values.ravel())[:N], distances.shape)[0]

    return indices

def classify_percentiles(
        dataset: xr.Dataset,
        var_name: str,
        percentile: float,
        dim: str,
        skipna: bool = True,
        dropna: bool = True,
) -> xr.Dataset:
    """
    Description:
        Classifies the given data into "1" for above percentile data (including the percentile) and 0 else.
    Parameters:
        dataset : Data for which the classification is done. 
        var_name : The datavariable of the dataset which should be classified
        percentile : Used to classify the data
        dim : The dimension of the data along which to calculate the percentile
        skipna : Whether to skip NaN values when calculating the percentile. Should be set True in most cases.
        dropna : Whether to drop NaN values at the beginning of the classification.
    Returns:
        classified_data : Classified data along the given dimension (dim).
    """

    data_ = dataset[var_name].copy() 

    # Drop NaN values of data if dropna is True
    if dropna:
        data_ = data_.where(
            cond=data_.notnull(),
            drop=True,
        )

    # Compute Quantile, skip NaN values if skipna is True
    if skipna:
        quantile = data_.where(
            cond=~np.isnan(data_)).quantile(
            q=percentile,
            dim=dim,
            skipna=skipna,
        )
    else:
        quantile = data_.quantile(
            q=percentile,
            dim=dim,
            skipna=skipna,
        )

    # Create the filter for classification
    above_percentile = data_ >= quantile
    below_percentile = data_ < quantile

    # Classify data (One Hot Encoding)
    classified_data = xr.where(
        cond=above_percentile,
        x=1,
        y=np.nan,
    )
    classified_data = xr.where(
        cond=below_percentile,
        x=0,
        y=classified_data,
    )

    classified_data[f'{var_name}_percentile'] = quantile

    classified_data = classified_data.rename(f"classified_{var_name}") # To avoid conflict when merging
    classified_data = xr.merge([dataset, classified_data])

    return classified_data

def classify_percentiles_da(
        data : xr.DataArray,
        percentile : float,
        dim : str,
        skipna : bool = True,
        dropna : bool = True,
) -> xr.DataArray:
    """
    Description:
        Classifies the given data into "1" for above percentile data (including the percentile) and 0 else.
    Parameters:
        data : Data for which the classification is done. 
        Needs to have only the time dimension if it is a timeseries
        percentile : Used to classify the data
        dim : The dimension of the data along which to calculate the percentile
        skipna : Whether to skip NaN values when calculating the percentile
        dropna : Whether to drop NaN values at the beginning of the classification.
    Returns:
        classified_df : Classified data along the given dimension (dim).
        Note that the dimension is reduced by NaN values if dropna = True. 
    """
    data_ = data.copy()

    # Drop NaN values of data
    data_ = data_.where( 
            cond = data_.notnull(),
            drop = dropna,
        )
    
    # Compute Quantile
    quantile = data_.quantile(
        q=percentile, 
        dim=dim,
        skipna=skipna,
        )
    
    # Create the filter for classification
    above_percentile = data_ >= quantile
    below_percentile = data_ < quantile

    # Classify data (One Hot Encoding)
    classified_df = data_.where(
    cond = below_percentile,
    other = 1,
    ).where(
        cond = above_percentile,
        other = 0
    )

    return classified_df

def poly_detrend(
        dataset: xr.Dataset,
        dv: str,
        dim: str,
        group_dim: str=None,
        deg: int=1, 
        ):
    """
    Description:
        Detrends a given data variable for each group in group_dim along the dim dimension.
    Parameters:
        dataset (xr.Dataset): The input xarray dataset.
        dv (str): The name of the datavariable to detrend
        dim (str): Detrend data along this dimension
        group_dim (str): Detrend for each subset of this dimension. 
                        If None the dataset should only have one dimension (dim).
        deg (int): The degree of polynomial interpolation
    Returns:
        dataset (xr.Dataset): The detrended dataset with new datavariables: detrended and trend
    """
    
    data_var = dataset[dv]

    # Iterate over each station
    if group_dim:
        # Initialize empty datasets to store detrended data and trends
        trend_da = xr.DataArray(
                data = np.empty((len(dataset[group_dim]), len(dataset[dim]))), # Needed to avoid the entries of trend_da to be of type object
                dims = [group_dim, dim],
                coords = {
                    group_dim: dataset[group_dim].values, 
                    dim: dataset[dim].values,
                    },
            )
        
        detrend_da = xr.DataArray(
                data = np.empty((len(dataset[group_dim]), len(dataset[dim]))), # Needed to avoid the entries of trend_da to be of type object
                dims = [group_dim, dim],
                coords = {
                    group_dim: dataset[group_dim].values, 
                    dim: dataset[dim].values,
                    },
            )
        for group in dataset[group_dim].values:
            # Select data for the current station
            group_data = data_var.sel({group_dim : group})

            # Get coefficients of polyfit for detrending
            polyfit_coeff = group_data.polyfit(dim=dim, deg=deg, skipna=True)

            # Compute polynomial for the trend
            trend = xr.polyval(group_data[dim], polyfit_coeff)

            # Subtract trend directly from the dataset
            detrended_data = dataset[dv].loc[{group_dim: group}] - trend
            detrend_da.loc[{group_dim: group}] = detrended_data["polyfit_coefficients"].values

            # Store detrended data and trend
            trend_da.loc[{group_dim: group}] = trend["polyfit_coefficients"].values
            
        # Create new Datavariables in the dataset 
        dataset["trend"] = trend_da
        dataset["detrended"] = detrend_da
    else:
        # Compute polynomial fit coefficients
        polyfit_coeff = data_var.polyfit(dim=dim, deg=deg, skipna=True)

        # Compute polynomial for the trend
        trend = xr.polyval(data_var, polyfit_coeff)

        # Detrend the data
        detrend_da = data_var - trend
        detrend_da = detrend_da["polyfit_coefficients"].rename("detrended")
        trend_da = trend["polyfit_coefficients"].rename("trend")

        # Store detrended data and trend
        dataset = xr.merge([dataset, detrend_da, trend_da])

    return dataset

def get_daily_max(
        data : xr.DataArray,
        dv_name: str,
        dim : str,
) -> xr.DataArray:
    """
    Description:
        Reduces hourly data to daily data and returns the maximum per day.
    Parameters:
        data : The (hourly) data to calculate the maximum of
        dv_name (str): Name of the DataArray to calculate the maximum of
        dim : The time dimension to convert to daily data
    Returns:
        daily_max : The daily maxima
    """
    # Convert to daily data
    data = data[dv_name]
    _data = data.copy()
    _data[dim] = _data[dim].dt.floor('D')

    # Take maximum for each day
    daily_max = _data.groupby(dim).max()
    daily_max = daily_max.rename(f"dmax_{dv_name}")
    
    return daily_max

def intersect1d(
        da1 : xr.DataArray,
        da2 : xr.DataArray,
        dim1 : str,
        dim2 : str,
) -> tuple : 
    """
    Description:
    Parameters:
        da1, da2 : The arrays to intersect
        dim1, dim2 : Dimensions of the arrays to intersect
    Returns:
        intersected_data : Tuple containing both xr.DataArrays with the intersected data.
    """
    da1_time = da1[dim1].values
    da2_time = da2[dim2].values

    intersected_time = np.intersect1d(da1_time, da2_time)

    intersected_da1 = da1.sel({
        dim1 : intersected_time
    })

    intersected_da2 = da2.sel({
        dim2 : intersected_time
    })

    intersected_data = (intersected_da1, intersected_da2)
    return intersected_data

def bandpass_filter(
        ds : xr.Dataset,
        group_dim : str,
        dv_name : str,
        p_high : float,
        p_low : float,
        fs : float,
        order : int,
        nan_handle: str,
    ):
    """
    Description: 
        Applies a bandpass-filter on the data for each group of the group_dim. 
        Returns the ds as it is but adds two data variables "signal" and "bandpass_filtered" to it.
    Parameters:
        ds : Dataset that contains the data to apply the filter on
        group_dim : The name of the dimension along which the data should be grouped
        dv_name : The name of the data array with the values of interest for applying the filter
        p_high : Upper threshold of the period to cut out (must be in the same timesteps as fs)
        p_low : Lower threshold of the period to cut out (must be in the same timesteps as fs)
        fs : Sampling frequency
        order : filter order
        nan_handle (str): Either "interp" or "zeros". Either interpolates the NaN values or sets them to zero.

    Returns:
    """
    #---
    # Auxillary Functions
    #---

    lowcut = 1 / (p_high * 2)  # Cutoff frequency for the lower end (1/period)
    highcut = 1 / (p_low / 2)  # Cutoff frequency for the higher end (1/period)

    # Define the function to apply the bandpass filter
    def apply_bandpass_filter(
            # TODO
            data, 
            lowcut, 
            highcut, 
            fs, 
            order, 
            nan_handle="interp",
            ):
        
        # Handle NaN values
        if nan_handle == "interp":
            mask_nan = np.isnan(data)
            data_interp = data.copy()
            data_interp[mask_nan] = np.interp(
                np.flatnonzero(mask_nan), 
                np.flatnonzero(~mask_nan), 
                data[~mask_nan])
            fdata = data_interp.copy()

        elif nan_handle == "zeros":
            fdata = data.copy()
            fdata[np.isnan(fdata)] = 0
            
        else: #TODO: Throw Error
            print("This Nan Handle does not exist")
            return None

        # Apply filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, fdata, axis=0)
        
        return filtered_data
    
    # Iterate over each station in the dataset
    filtered_signals = []
    for group in ds[group_dim]:
        # Extract data for the current station
        group_data = ds[dv_name].sel({group_dim : group}).values
        
        # Apply bandpass filter to the interpolated data
        filtered_data = apply_bandpass_filter(
            group_data, 
            lowcut, highcut, 
            fs, 
            order,
            nan_handle,
        )        
        # Store the filtered signal
        filtered_signals.append(filtered_data)

    # Create a new xarray Dataset with the filtered signal and the difference from the original data
    filtered_signals = np.array(filtered_signals)
    signal_variable = xr.DataArray(
        filtered_signals,
        dims=ds[dv_name].dims,  # Use the dimensions from the detrended variable in ds
        coords={dim: ds[dim] for dim in ds[dv_name].dims} # Use the coordinates from ds for each dimension)
        )  

    filtered_ds = xr.Dataset({'signal': signal_variable})

    # Calculate the bandpass filtered data by subtracting the signal from the detrended data
    filtered_ds['bandpass_filtered'] = ds[dv_name] - filtered_ds["signal"]

    # Combine the original dataset with the filtered data
    filtered_ds = xr.merge([ds, filtered_ds])

    # Now filtered_ds contains the original data along with the bandpass filtered signal and the difference,
    # preserving NaN values in the output dataset.

    return filtered_ds

def filter_years_months(ds, years=None, months=None, dim_name='date_time'):
    """
    Filter the given xarray dataset `ds` based on the specified list of years and months.
    
    Parameters:
        ds (xarray.Dataset): The input xarray dataset.
        years (list): A list of years to filter.
        months (list): A list of months (1-12) to filter.
        dim_name (str): Name of the dimension to filter along.
        
    Returns:
        xarray.Dataset: The filtered dataset.
    """
    # Select data for the specified years and months
    if months is not None:
        ds = ds.sel(**{dim_name: ds[f'{dim_name}.month'].isin(months)})
    if years is not None:
        ds = ds.sel(**{dim_name: ds[f'{dim_name}.year'].isin(years)})

    filtered_data = ds 
     
    return filtered_data

def detrend_with_xr(
        data:xr.DataArray,
        dim:str,
        deg:int,
        skipna:bool = False,
        ):
    """
    Description: TODO: OLd version. New one is poly_detrend which can get a dataset as input
    Parameters:
    Returns:
    """
    df = data.copy()
    
    # Get coefficients of polyfit
    polyfit_coeff = df.polyfit(
    dim=dim, 
    deg=deg, 
    skipna=skipna,
    )

    # Compute polynomial
    polynomials = xr.polyval(
    coord=df.dropna(dim)[dim],
    coeffs=polyfit_coeff,
    )
    polynomials = polynomials.rename({'polyfit_coefficients': 'polyfit'}) # Contains now the polynomial

    # Subtract trend from data ignoring NaN values
    time_to_update = df.dropna(dim)[dim] # Select timepoints with measurements
    df.loc[{dim: time_to_update}] -= polynomials["polyfit"].values

    return df, polynomials