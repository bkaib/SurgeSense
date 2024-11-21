#---
# Modules
#---
import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
#---
# Graphical evaluation
#---
def plot_cf(model, X_test, y_test,):
    """
    Description:
        Plots a 2x2 Confusion Matrix labeled with rates and absolute values

    Parameters:
        model (RandomForestClassifier): Fitted model (RF).
        X_test (np.array): Predictors of test data. Shape(time, features)
        y_test (np.array): Predictand of test data. Shape(time,)

    Returns:
        fig (matplotlib.figure.Figure): Figure of labeled confusion matrix. Percentages are rates (False Positive Rate etc.)
    
    Source (adjusted):
        "https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/"
    """
    
    # Plot Confusion Matrix
    #---
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Get confusion matrix
    y_test_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_test_pred)

    # Labels for quadrants in matrix
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    # Calculate Rates of Matrix
    tn, fp, fn, tp = cf_matrix.ravel()
    tnr = tn / (tn + fp) # True negative rate
    fpr = 1 - tnr # False Positive Rate
    tpr = tp / (tp + fn) # True positive rate
    fnr = 1 - tpr # False negative rate
    values = [tnr, fpr, fnr, tpr,]

    rates = ["{0:.2%}".format(value) for value in values]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names, group_counts, rates)]

    labels = np.asarray(labels).reshape(2,2)

    # Main Figure
    fig, ax = plt.subplots(tight_layout=True)
    
    ax = sns.heatmap(cf_matrix,
    annot=labels, 
    fmt='', 
    cmap='Blues',
    )

    # Axis labels
    ax.set_title('Confusion Matrix \n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Axis ticks
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

    return fig

def importance_map(importance, lons, lats, tflag="", ax=None):
    """
    Description:
        Plots importance of a predictor on a lat lon map.
    Parameters:
        importance (np.array): Importances from model classification. Shape:(features,)
        lons (np.array): Longitudes of predictor importance
        lats (np.array): Latitudes of predictor importance
        tflag (str): Additional Title information
    Returns:
        fig: Figure of importance
        ax: Axis to draw position of station into it
    """
    # Modules
    import matplotlib.pyplot as plt 
    import cartopy.crs as ccrs

    # Reshape importance to lat/lon
    nlat = len(lats)
    nlon = len(lons)
    importance = importance.reshape(nlat, nlon)

    # Adjust colorbar extremes
    vmax = 0.5 * np.max(importance)

    # Plot importance on lat/lon map
    fig = plt.figure(tight_layout=True,)

    if ax == None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        
    plot = ax.contourf(lons, lats, importance, 
    transform=ccrs.PlateCarree(),
    vmax=vmax,
    )

    ax.coastlines()

    plt.colorbar(plot,)

    ax.set_title(f"Importance {tflag}")

    return fig, ax

def overlay_importance(ax, importance, lats, lons, percentile=99, alpha=0.1, markersize=5, color="k"):
    # TODO: put into module
    """
    Description:
        Overlays the position of importance above a specified percentile to an axis.
        Mainly this is used to overlay the importance on predictor maps.
    Parameters:
        ax
        importance
        lats
        lons:
        percentile (float): Percentile between 0. and 100.
    Returns:
    """
    import matplotlib.lines as mlines

    # Reshape importance
    #---
    nlat = len(lats)
    nlon = len(lons)
    importance = importance.reshape(nlat, nlon)

    # Find positions above percentile
    #---
    perc = np.percentile(importance, q=percentile)
    perc_lat_idx, perc_lon_idx = np.where(importance > perc) 
    perc_lats = lats[perc_lat_idx]
    perc_lons = lons[perc_lon_idx]

    # Plot pairs into figure
    #---
    ax.plot(perc_lons, perc_lats, "s", markersize=markersize, alpha=alpha, color=color, transform=ccrs.PlateCarree())

#---
# Metrics
#---
def cfm_metrics(model, X_test, y_test, beta=0.5):
    """
    Description:
        Use standard metrics for random forest classification with confusion matrix.
    Parameters:
        model (RandomForestClassifier): Fitted model (RF).
        X_test (np.array): Predictors of test data. Shape(time, features)
        y_test (np.array): Predictand of test data. Shape(time,)
        beta (float): Weight for measure of weighted accuracy. (Default:0.5) 
    Returns:
        metrics (dict): Set of metrics.
    """
    # Modules
    from sklearn.metrics import confusion_matrix

    # Confusion Matrix
    y_test_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_test_pred)

    # Calculate Metrics
    tn, fp, fn, tp = cf_matrix.ravel()
    tnr = tn / (tn + fp) # True negative rate
    tpr = tp / (tp + fn) # True positive rate
    gmean = np.sqrt(tnr * tpr) # G-Mean
    beta = 0.5 # Weight
    wacc = beta * tpr + (1 - beta) * tnr # Weighted Accuracy
    precision = tp / (tp + fp) 
    recall = tpr
    fmeasure = (2 * precision * recall) / (precision + recall) # F-measure

    # Display metrics
    names = ("tnr", "tpr", "gmean", "wacc", "precision", "recall", "fmeasure")
    values = (tnr, tpr, gmean, wacc, precision, recall, fmeasure)
    metrics = dict(zip(names, values))

    print("Metric values \n")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

def relative_scores(trainscore, testscore):
    """
    Description:
        Indicates if model has possibly over- or underfitted based on the relative
        rate of train- and testscores.
    Parameters:
        trainscore (float): Trainscore of model
        testscore(float): Testscore of model
    Returns:
        Possible overfit (underfit) if trainscore is at least 10% higher (lower) than testscore
        relative_fit (float): trainscore / testscore
    """
    import numpy as np

    #- Calculate ratio
    relative_fit = trainscore / testscore  

    #- Print scores
    print(f"testscore: {testscore}")
    print(f"trainscore: {trainscore}")
    print(f"train/test: {relative_fit}")

    #- Evaluate over- and underfitting
    if (relative_fit) >= 1.1: # train score 10% higher than test score
        print("Possible overfit")
    if (relative_fit) <= 0.9: # train score 10% lower than test score
        print("Possible underfit")

    return relative_fit
#---
# Auxillary Functions
#---
def separate_predictors(X, n_pred_features):
    """
    Description: 
        Separates a single predictor from all features when multiple predictors were passed to the model
    Parameters:
        X (np.array): Predictor values of all features passed to model, Shape:(n_predictions, n_features,)
        n_pred_features (int): number of features of a single predictor within all features
    Returns:
        predictor_importance (np.array): Importance of a single predictor from a model run, Shape:(n_predictors, n_pred_features)
    Note: 
        From rf009 combinations of predictors and timelags are allowed. This function orders the n_pred_features as follows: 
        1: timelag1,pred1, 2:timelag1,pred2, ..., n: timelag1,pred_n, n+1: timelag2, pred1, ... t*n: timelag_t, pred_n
        For an example see notebooks>rf009.ipynb
    """

    # Separate importance of each predictor
    separated_predictors = []
    n_features = X.shape[1]
    n_predictors = n_features // n_pred_features

    start = 0
    for i in range(n_predictors):
        end = start + n_pred_features 
        single_predictor = X[:, start : end]
        start = end # Update index for next predictor
        
        separated_predictors.append(single_predictor)

    separated_predictors = np.array(separated_predictors, dtype=object) 

    return separated_predictors

def separate_predictor_importance(importance, n_pred_features):
    """
    Description: 
        Separates the importance of a single predictor from all features when multiple predictors were passed to the model
    Parameters:
        importance (np.array): Importance values of all features passed to model, Shape:(n_features,)
        n_pred_features (int): number of features of a single predictor within all features
    Returns:
        predictor_importance (np.array): Importance of a single predictor from a model run, Shape:(n_predictors, n_pred_features)
    Note: 
        From rf009 combinations of predictors and timelags are allowed. This function orders the n_pred_features as follows: 
        1: timelag1,pred1, 2:timelag1,pred2, ..., n: timelag1,pred_n, n+1: timelag2, pred1, ... t*n: timelag_t, pred_n
        For an example see notebooks>rf009.ipynb
    """

    # Separate importance of each predictor
    n_features = len(importance)
    n_predictors = n_features // n_pred_features # ERA5
    n_pfs = n_features % n_pred_features # Prefillings
    
    predictor_importance = []

    # Loop over all ERA5 Predictors and collect their importance values
    start = 0
    for i in range(n_predictors):
        end = start + n_pred_features 
        pred_importance = importance[start : end]
        start = end # Update index for next predictor
        
        predictor_importance.append(pred_importance)

    # Loop over prefilling values and collect their importance values
    if n_pfs > 0:
        prefilling_values = importance[-n_pfs:] # Last values are prefilling values
        for value in prefilling_values:
            predictor_importance.append(value)
    predictor_importance = np.array(predictor_importance, dtype=object)
    # #####
    # # Old
    # ####
    # predictor_importance = []
    # n_features = len(importance)
    # n_predictors = n_features // n_pred_features

    # start = 0
    # for i in range(n_predictors):
    #     end = start + n_pred_features 
    #     pred_importance = importance[start : end]
    #     start = end # Update index for next predictor
        
    #     predictor_importance.append(pred_importance)

    # predictor_importance = np.array(predictor_importance, dtype=object) # If number of lon lats is not the same (can this happen?)

    return predictor_importance