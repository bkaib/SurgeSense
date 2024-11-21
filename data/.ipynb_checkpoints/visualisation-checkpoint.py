#--- 
# Modules
#---
import numpy as np
from models import evaluation
from data import saver
import matplotlib.pyplot as plt
from data import preprocessing

def pred_map(data, lons, lats, vmin=None, vmax=None, tflag="", unit=None, is_clb_label=True, nlevels=6, cmap="viridis", pad=None):
    """
    Description:
        Plots predictor-data on a lat-lon map.
    Parameters:
        data (np.array): dataset from predictors. Shape:(lat, lon).flatten()
        lons (np.array): Longitudes of predictor data
        lats (np.array): Latitudes of predictor data
        tflag (str): Additional Title information, (Defaults: "")
        vmin (float): Minimum value for colorbar, (Defaults: None)
        vmax (float): Maximum value for colorbar, (Defaults: None)
        unit (str): Unit of data
        is_clb_label (bool): Whether to label the colorbar with unit or not.
        nlevels (int): Number of levels in contour plot (e.g. sections of the colorbar)
        cmap (str): Colormap of the plot
    Returns:
        fig: Figure of data
    """
    # Modules
    import matplotlib.pyplot as plt 
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import numpy as np

    # Reshape data to lat/lon
    nlat = len(lats)
    nlon = len(lons)
    data = data.astype(float)
    data = data.reshape(nlat, nlon)

    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    
    # Plot data on lat/lon map
    fig = plt.figure(tight_layout=True,)
    ax = plt.axes(projection=ccrs.PlateCarree())

    plot = ax.contourf(lons, lats, data,
    levels=np.linspace(vmin, vmax, num=nlevels),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    extend='both',
    )

    ax.coastlines()
    
    num_cbar_ticks = round(0.5 * nlevels)
    if vmin == None or vmax == None:
        cbarticks = None
    else:
        cbarticks = np.linspace(vmin, vmax, num=num_cbar_ticks)

    # clb = plt.colorbar(plot, shrink=.62, ticks=cbarticks,)
    clb = plt.colorbar(plot, ticks=cbarticks)

    if is_clb_label:
        clb.set_label(f"{unit}", rotation=90, labelpad=1)

    ax.set_title(f"{tflag}", pad=pad)

    ax.set_xticks(lons[::10][::4], crs=ccrs.PlateCarree()) # Deg E
    ax.set_yticks(lats[::15][::2], crs=ccrs.PlateCarree()) # Deg N

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.close()

    return fig, ax

def create_gif(figures, path, fps=1):
    """
    Description: 
        Creates and saves a gif from a list of figures
    Parameters:
        figures (list): List of Figures with 2 Axes.
        path (str): Path to save gif to.
        fps (int): Frames per second for gif. (Defaults: 1)
    Returns: 
        None
    """
    import imageio
    import numpy as np
    images = []
    for fig in figures:
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8') # Convert to RGB values
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Reshape for gif

        images.append(image)

    imageio.mimsave(f'{path}.gif', images, fps=fps)

def predictor_maps(model, X_test, y_test, X_test_unscaled, ndim, n_pfs, is_pf_combined, lons, lats, pred_units, pred_names, station_positions, station_names, is_station_name, is_overlay_importance, run_id, model_run, percentile, markersize, alpha, color, colorbar_range, nlevels=10):
    """
    Description:
        Plots values of all predictors used for model training. Selects timepoints where Storm Surges were in original data and indicates whether 
        the prediction was true or not in the filename ("isfalse", "istrue"). The file naming "predss" is for situations, where a storm surge was predicted
        but the original data has no storm surge.
    Parameters:
        model (clf): Model that was fitted to X_test, y_test.
        X_test (): Test set of predictor data used for model fit
        y_test (): Test set of predictand data used for model fit
        lons (): Values of longitudes of predictors
        lats (): Values of latitudes of predictors
        pred_units (list): Units of all predictors used, e.g. ms**-1
        pred_names (list): Names of all predictors used, e.g. sp_tlag0
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
        station_name (str): Name of the station given in GESLA dataset.
        is_station_name (bool): Whether to indicate station name in plot or not (Defaults: False)
        is_overlay_importance (bool): Overlay values of importance of that predictor or not
        run_id (int): Number of the model run
        model_run (str): Name of the current model run
        percentile (float): Percentile (0-100) for selecting position of overlayed importance
        markersize (int): Size of square-marked positions of importance
        alpha (float): Transparency of square-markers for importance
        color (str): Color of square-marker for importance
        colorbar_range (dict): Keys are predictors (e.g. "sp"), values are list of [vmin, vmax] values for colorbar
        nlevels (int): Number of levels in contour plot (e.g. sections of the colorbar) (Defaults: 10)

    Returns:
        None
    """
    #---
    # Make a prediction
    #---
    nlat = lats.size
    nlon = lons.size
    y_test_pred = model.predict(X_test) # Predicted data
    importance = model.feature_importances_
    n_pred_features = nlon * nlat # Features per predictor (lon/lat Input-Field). Needed for importance separation

    # Get importance per predictor of model
    #---
    if is_pf_combined:
        importance = importance[:-n_pfs] # Select only values that contain era5 as predictor
        X_test = X_test[:, :-n_pfs]
        X_test_unscaled = X_test_unscaled[:, :-n_pfs]

    predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features) #

    #---
    # Select data for plotting original storm surge events
    #---
    ss_idx = np.where(y_test == 1) # Timepoints of storm surges in original data

    y_test_ss = y_test[ss_idx] # Original data storm surges only
    y_pred_ss = y_test_pred[ss_idx] # Predictions at timepoints of SS in original data

    ntime = X_test.shape[0]
    X_pred = X_test_unscaled.reshape(ntime, -1, nlat, nlon) # Reshape to fit format for plotting predictor values on a map TODO: Needs to be adjusted to unscaled values?

    X_pred_plot = X_pred[ss_idx] # Select only predictor values at timepoints of storm surges

    #---
    # Plot & Save predictor map at original storm surge events
    #---
    n_time = X_pred_plot.shape[0]
    n_pred = X_pred_plot.shape[1]

    time_idx = 0
    for time in range(n_time):

        is_correct_prediction = (y_test_ss[time] == y_pred_ss[time])

        for pred_idx in range(n_pred):
            pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
            # Convert unit of colorbar
            #---
            unit = pred_units[pred_idx]
            if (unit == "m s**-1"): 
                unit = "m/s"

            # Create Figure
            #---
            data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

            if is_correct_prediction:
                tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 1"
                fname = f"{pred_names[pred_idx]}_{time_idx}_istrue_{run_id}"
            else:
                tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 0" 
                fname = f"{pred_names[pred_idx]}_{time_idx}_isfalse_{run_id}"
            
            # Get colorbar vmin vmax
            #---
            vmin = colorbar_range[pred_flag][0] 
            vmax = colorbar_range[pred_flag][1]

            # Choose colormap
            #---
            if pred_flag == "msl":
                cmap = "coolwarm"
            elif pred_flag == "tp":
                cmap = "Blues"
            elif (pred_flag == "u10" or pred_flag == "v10"):
                cmap= "seismic"

            # Plot figure
            #---
            fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)
            
            # Add position of station to map
            #---
            for station_name in station_names:
                plot_station(ax, station_positions, station_name, is_station_name)

            # Add importance to map
            #---
            if is_overlay_importance:
                pred_importance = predictor_importances[pred_idx]
                evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)

            # Save plot & data
            #---
            folder1 = f"results/random_forest/{model_run}/predictor_maps/"
            saver.directory_existance(folder1)
            np.save(file=f"{folder1}{fname}_data", arr=data)
            np.save(file=f"{folder1}{fname}_importance", arr=pred_importance)
            fig.savefig(f"{folder1}{fname}.pdf")

        time_idx = time_idx + 1

    #---
    # Plot & Save predictor map of predicted storm surge events where original data has no storm surge 
    #---
    idx2 = np.where(y_test_pred == 1) # Select all occurences where prediction has SS
    y_test_idx2 = y_test[idx2]
    X_pred_plot = X_pred[idx2]
    idx3 = np.where(y_test_idx2 == 0) # Subselect all occurences where prediction has SS and original data has no SS
    X_pred_plot = X_pred_plot[idx3] # Choose this selection as a plot
    n_time = X_pred_plot.shape[0]
    n_pred = X_pred_plot.shape[1]

    for time in range(n_time):
        for pred_idx in range(n_pred):
            pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
            # Convert unit of colorbar
            #---
            unit = pred_units[pred_idx]
            if (unit == "m s**-1"): 
                unit = "m/s"

            # Create Figure
            #---
            data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

            tflag = f"{pred_names[pred_idx]},  y_orig = 0, y_pred = 1"
            
            # Get colorbar vmin vmax
            #---
            vmin = colorbar_range[pred_flag][0] # Split because string is "tp_tlag0" etc.
            vmax = colorbar_range[pred_flag][1]

            # Choose colormap
            #---
            if pred_flag == "sp":
                cmap = "coolwarm"
            elif pred_flag == "tp":
                cmap = "Blues"
            elif (pred_flag == "u10" or pred_flag == "v10"):
                cmap= "seismic"
                
            # Plot figure
            #---
            fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)

            # Add position of station to plot
            #---
            for station_name in station_names:
                plot_station(ax, station_positions, station_name, is_station_name)

            # Add importance to map
            if is_overlay_importance:
                pred_importance = predictor_importances[pred_idx]
                evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)
            
            # Save figure
            #---
            folder1 = f"results/random_forest/{model_run}/predictor_maps/"
            saver.directory_existance(folder1)

            fname = f"{pred_names[pred_idx]}_{time_idx}_predss_{run_id}"

            fig.savefig(f"{folder1}{fname}.pdf")

        time_idx = time_idx + 1

def plot_station(ax, station_positions, station_name, is_station_name=False, markersize=8, fontsize=12, color="k", is_legend=True):
    """
    Description:
        Plots the position of a station into given axis
    Parameters:
        ax (GeoAxesSubplot): Axis to plot station into
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
        station_name (str): Name of the station given in GESLA dataset.
        is_station_name (bool): Whether to indicate station name in plot or not (Defaults: False)
        
        kwargs**
        markersize (float): Size of the cross used to indicate position of station (Defaults: 10)
        fontsize (float): Fontsize of the name of station (Defaults: 12)
        color (str): Color code (Defaults: "r")
    Returns:
        None
    """
    # import matplotlib.pyplot as plt 
    import matplotlib.lines as mlines
    import cartopy.crs as ccrs

    lons = station_positions[station_name][0]
    lats = station_positions[station_name][1]

    # Mark position of station
    ax.plot(lons, lats, 'X', markersize=markersize, color=color, transform=ccrs.PlateCarree())

    # Add legend
    if is_legend:
        black_cross = mlines.Line2D([], [], color=color, marker='X', linestyle='None',
                            markersize=markersize, label=station_name)

        ax.legend(handles=[black_cross,], loc="lower center", bbox_to_anchor=(0., 0., 0.5, 0.5), fontsize=8,)

    if is_station_name:
        ax.text(1.001*lons, 1.001*lats, station_name, fontsize=fontsize, transform=ccrs.PlateCarree())

def generate_pie_chart(
        data: np.array,
        labels: list,
        title: str="Pie Chart",
        colors: list=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#fdbf6f', '#80b1d3'],
):
    """
    Description:
        Visualizes the given data and labels as a pie-chart.
    Parameters:
    Returns:
    """
    n = len(labels)
    # Define pastel colors
    colors = colors[:n]

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.pie(
        data, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors)  # autopct enables you to display the percent value using Python string formatting
    plt.title(title)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Return the figure
    return plt.gcf()

def visualize_importance_old(model, n_pred_features, selected_predictors, timelags):
      
    importance = model.feature_importances_

    ## Separate importance per predictor
    #---
    separated_importance = evaluation.separate_predictor_importance(
        importance=importance,
        n_pred_features=n_pred_features,
    )

    ## Populate a dict for saving the data
    #---
    importance_dict = {}
    pie_labels = []
    for i, predictor in enumerate(selected_predictors):
        importance_dict[predictor] = separated_importance[i, :]
        pie_labels.append(f"{predictor}_tlag{timelags[i]}")

    ## Compute importance per predictor
    #---

    aggregated_importance = np.sum(separated_importance, axis=1)

    #---
    # Visualize in a pie-plot
    #---
    pie_fig = generate_pie_chart(
        data=aggregated_importance, 
        labels=pie_labels,
        title="Distribution of Predictor Importance",
        )
    return pie_fig, importance_dict

def visualize_importance(model, n_pred_features, selected_predictors, timelags):
  
    importance = model.feature_importances_
    
    ## Separate importance per predictor
    #---
    separated_importance = evaluation.separate_predictor_importance(
        importance=importance,
        n_pred_features=n_pred_features,
    )
    
    ## Populate a dict for saving the data
    #---
    importance_dict = {}
    pie_labels = []
    for i, predictor in enumerate(selected_predictors):
        importance_dict[predictor] = separated_importance[i,]
        pie_labels.append(f"{predictor}_tlag{timelags[i]}")
    
    ## Compute importance per predictor
    #---
    aggregated_importance = []
    for importance in separated_importance:
        aggregated_importance.append(np.sum(importance))
    aggregated_importance = np.array(aggregated_importance)    
    #---
    # Visualize in a pie-plot
    #---
    pie_fig = generate_pie_chart(
        data=aggregated_importance, 
        labels=pie_labels,
        title="Distribution of Predictor Importance",
        )
    return pie_fig, importance_dict
    
def mean_pms(
        model: str,
        y_test: np.array,
        X_test: np.array,
        X_test_unscaled: np.array,
        predictors: list,
        timelags: list,
        pred_units: list,
        lons: np.array,
        lats: np.array,
        cmap: dict,
        colorbar_range: dict,
        nlevels: int,
        station_name: str,
        station_positions: dict,
        model_run_code: str,
        run_id: int,
        save_id: str=None,
        ):
    """
    Returns a dict: keys are predictors
    """
    #---
    # Get days of TPs, FPs, FNs
    #---
    y_test_pred = model.predict(X_test)
    n_pred_features = len(lons) * len(lats)
    importance = model.feature_importances_


    # Filter TP (3), FN (2), FP(1), TN(0)
    y_filtered = 2 * y_test + y_test_pred # Filter 
    tp_idx = np.where(y_filtered == 3)
    fn_idx = np.where(y_filtered == 2)

    X_test_pred = evaluation.separate_predictors(X_test_unscaled, n_pred_features) # Its not the importance but general separation

    X_test_tp = np.mean(X_test_pred[:, tp_idx[0], :], axis=1) # (4, 17061) e.g. (pred, nlat * nlon)
    X_test_fn = np.mean(X_test_pred[:, fn_idx[0], :], axis=1)
    
    for pred_idx, predictor in enumerate(predictors): 
        print(f"visualize curent predictor: {predictor}")
        if predictor == "pf":
            print("no maps for prefilling")
            pass # No maps for pf as a predictor
        else:
            #---
            # Plot figure
            #---
            unit = pred_units[pred_idx]           
            tlag = timelags[pred_idx]
    
            data1 = X_test_fn[pred_idx, :] - X_test_tp[pred_idx, :]
            data2 = X_test_tp[pred_idx, :]
            data3 = X_test_fn[pred_idx, :]
    
            vmin = min(data1)
            vmax = max(data1)
            tflag = f"""Difference of mean predictor maps 
            for {predictor} with timelag {tlag}"""
    
            fig1, ax1 = pred_map(data1, 
                                lons, lats, 
                                tflag=tflag, 
                                unit=unit, 
                                vmin=vmin, vmax=vmax, 
                                nlevels=nlevels, 
                                cmap=cmap[predictor], 
                                pad=None)
            
            vmin, vmax = colorbar_range[predictor]
            tflag = f"""Mean true positive predictor map
            for {predictor} with timelag {tlag}"""
    
            fig2, ax2 = pred_map(data2, 
                                lons, lats, 
                                tflag=tflag, 
                                unit=unit, 
                                vmin=vmin, vmax=vmax, 
                                nlevels=nlevels, 
                                cmap=cmap[predictor], 
                                pad=None)
    
            vmin, vmax = colorbar_range[predictor]
            tflag = f"""Mean false negative predictor map
            for {predictor} with timelag {tlag}"""
    
            fig3, ax3 = pred_map(data3, 
                                lons, lats, 
                                tflag=tflag, 
                                unit=unit, 
                                vmin=vmin, vmax=vmax, 
                                nlevels=nlevels, 
                                cmap=cmap[predictor], 
                                pad=None)
            
            # Add position of station to map
            #---
            plot_station(ax1, station_positions, station_name, is_station_name=False, is_legend=False)
            plot_station(ax2, station_positions, station_name, is_station_name=False, is_legend=False)
            plot_station(ax3, station_positions, station_name, is_station_name=False, is_legend=False)
    
            # Add importance to map
            #---
            predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features)
            pred_importance = predictor_importances[pred_idx]
    
            evaluation.overlay_importance(ax1, pred_importance, lats, lons, percentile=99, alpha=0.08, markersize=5, color="k")
            evaluation.overlay_importance(ax2, pred_importance, lats, lons, percentile=99, alpha=0.08, markersize=5, color="k")
            evaluation.overlay_importance(ax3, pred_importance, lats, lons, percentile=99, alpha=0.08, markersize=5, color="k")
    
            
            # Save figure & Data
            #---
            folder = f"results/random_forest/{model_run_code}/predictor_maps/"
            saver.directory_existance(folder)
            
            if save_id is None:
                fname1 = f"meanmap_diff_{predictor}_tlag{tlag}_{run_id}"
                fname2 = f"meanmap_TP_{predictor}_tlag{tlag}_{run_id}"
                fname3 = f"meanmap_FN_{predictor}_tlag{tlag}_{run_id}"
            else:
                fname1 = f"{save_id}_meanmap_diff_{predictor}_tlag{tlag}_{run_id}"
                fname2 = f"{save_id}_meanmap_TP_{predictor}_tlag{tlag}_{run_id}"
                fname3 = f"{save_id}_meanmap_FN_{predictor}_tlag{tlag}_{run_id}"
    
            fig1.savefig(f"{folder}{fname1}.pdf")
            np.save(f"{folder}{fname1}", data1)
            print(f"Saved Figure to {folder}{fname1}")
    
            fig2.savefig(f"{folder}{fname2}.pdf")
            np.save(f"{folder}{fname2}", data2)
            print(f"Saved Figure to {folder}{fname2}")
    
            fig3.savefig(f"{folder}{fname3}.pdf")
            np.save(f"{folder}{fname3}", data3)
            print(f"Saved Figure to {folder}{fname3}")
        
def plot_cfm(model, X, y):
        # Plot Confusion Matrix
        #---
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        # Get confusion matrix
        y_pred = model.predict(X)
        conf_matrix = confusion_matrix(y, y_pred)

        # Labels for quadrants in matrix
        group_names = ['True Neg','False Pos','False Neg','True Pos']

        group_counts = ["{0:0.0f}".format(value) for value in
                        conf_matrix.flatten()]

        # Calculate Rates of Matrix
        tn, fp, fn, tp = conf_matrix.ravel()
        tnr = tn / (tn + fp) # True negative rate
        fpr = 1 - tnr # False Positive Rate
        tpr = tp / (tp + fn) # True positive rate
        fnr = 1 - tpr # False negative rate
        values = [tnr, fpr, fnr, tpr,]

        rates = ["{0:.2%}".format(value) for value in values]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names, group_counts, rates)]

        labels = np.asarray(labels).reshape(2,2)

        # Display confusion matrix with labels
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix', fontsize=16)
        plt.colorbar(ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues'))

        # Add labels from the "labels" array to the cells
        for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                        ax.text(j, i, labels[i, j], horizontalalignment='center', verticalalignment='center', color='black')

        ax.set_xlabel('Predicted Labels', fontsize=14)
        ax.set_ylabel('True Labels', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)
        ax.set_yticklabels(['Negative', 'Positive'], fontsize=12)
        plt.tight_layout()

        return fig

