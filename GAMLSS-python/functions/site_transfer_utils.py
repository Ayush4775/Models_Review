import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import numpy as np
import pandas as pd
import logging
import sys
import logging
import shutil
import os
import numpy as np
import pandas as pd
import pickle
from gamlass_main import compare_models_and_save
import multiprocessing as mp
from tqdm import tqdm
import pickle

gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')
utils = importr('utils')
grdevices = importr('grDevices')
from gamlass_main import *
from utils import *
from model_utils import *

import os
os.environ["R_LIBS_USER"] = "/usr/local/lib/R/site-library"



# Helper functions (these should be defined or imported)
# Get link functions for each parameter
def find_link(model_name, param):
    return robjects.r[model_name].rx2(f'{param}.link')[0]

# Function to create offset term based on link function
def create_offset_term(param, link, suffix="_pred"):
    param = param + suffix
    if link == "identity":
        return f"offset({param})"
    elif link == "log":
        return f"offset(log({param}))"
    elif link == "logit":
        return f"offset(log({param} / (1 - {param})))"
    elif link == "inverse":
        return f"offset(1 / {param})"
    elif link == "sqrt":
        return f"offset(sqrt({param}))"
    else:
        raise ValueError(f"Unsupported link function: {link}")


# Function to apply inverse link function
def apply_inverse_link(value, link):
    if link == "identity":
        return value
    elif link == "log":
        return np.exp(value)
    elif link == "logit":
        return 1 / (1 + np.exp(-value))
    elif link == "inverse":
        return 1 / value
    elif link == "sqrt":
        return value ** 2
    else:
        raise ValueError(f"Unsupported link function: {link}")
    
def get_coef(model_name, param):
    return robjects.r[model_name].rx2(f'{param}.coefficients')[0]


def adjust_predictions(predictions, gam_refit_name):
    adjusted_predictions = predictions.copy()
    available_params = robjects.r(f"{gam_refit_name}$parameters")
    param_names = list(available_params)

    for param in param_names:
        link = find_link(gam_refit_name, param)
        adjusted_predictions[f'{param}_pred'] += apply_inverse_link(get_coef(gam_refit_name, param), link)

    return adjusted_predictions

def calculate_z_score_transfer(adjusted_predictions, test_data, y_val, func_name):
    """
    Calculate Z-scores for the test data using adjusted predictions.

    Parameters:
    - adjusted_predictions (pandas.DataFrame): DataFrame containing the adjusted predictions.
    - test_data (pandas.DataFrame): DataFrame containing the test data.
    - y_val (str): Name of the target variable.
    - func_name (str): Distribution type (default is SHASH)

    Returns:
    - numpy.ndarray: Z-scores for the test data.
    """
    func_name = 'p' + func_name
    print(f"func_name: {func_name}")
    param_names = adjusted_predictions.columns
    
    test_data_name = "newdata"
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_test_data = robjects.conversion.py2rpy(test_data)
        robjects.globalenv[test_data_name] = r_test_data

        # param_names_clean = [name.replace('_pred', '') for name in param_names]
        # for i,param in enumerate(param_names):
        #     param_pred = adjusted_predictions[param].values
        #     #print(f"param: {param}, values: {param_pred}")
        #     robjects.globalenv[f'{param_names_clean[i]}'] = robjects.FloatVector(param_pred)
        
        # param_names_clean = [name.replace('_pred', '') for name in param_names]
        # param_inputs = ", ".join([f"{param} = {param}" for param in param_names_clean])

        for param in param_names:
            param_pred = adjusted_predictions[param].values
            robjects.globalenv[f'{param}'] = robjects.FloatVector(param_pred)
        
        param_names_clean = [name.replace('_pred', '') for name in param_names]
        param_inputs = ", ".join([f"{param} = {param}_pred" for param in param_names_clean])


        r_code = f"""
        m_quantiles <- {func_name}({test_data_name}${y_val}, {param_inputs})
        z_scores <- qnorm(m_quantiles, mean = 0, sd = 1)
        z_scores
        """
        #print(f"r_code: {r_code}")
        # Capture and suppress the R output
        z_scores = robjects.r(r_code)

    # Remove the assigned variables
    robjects.r(f'rm({test_data_name})')
    for param in param_names:
        robjects.r(f'rm({param})')

    return np.array(z_scores)



# Main site transfer function
def site_transfer(main_model_path,main_traindata, df2, new_site, selected_site, fraction, y_val, x_vals,iterations=100):
    """
    Perform site transfer for a single new site.
    
    Parameters:
    - main_model: The path of the main GAMLSS model
    - main_traindata: The training dataset for the main model (pandas DataFrame)
    - df2: The test dataset (pandas DataFrame)
    - new_site: Name of the new site (string)
    - selected_site: Name of the selected proxy site (string)
    - fraction: Fraction of data to use for adjustment (float)
    - y_val: Name of the target variable (string)
    - x_vals: List of predictor variable names (list of strings)
    - iterations: Number of iterations for the adjustment model (int)
    
    Returns:
    - predictions_df: Original predictions for the new site
    - adjusted_predictions: Adjusted predictions for the new site
    """

    # Loading the model
    main_model = Gamlss.load_model(main_model_path)
    
    # Prepare data for the new site
    columns = x_vals + [y_val]
    site_data = df2[df2['site'] == new_site]
    none_samples = site_data[site_data['affected'] == 'None']
    num_samples = int(len(none_samples) * fraction)
    site_sub_data = none_samples.sample(n=num_samples, random_state=42)
    train_set_index = list(site_sub_data.index)
    site_sub_data = site_sub_data.reset_index(drop=True)
    print(f"Number of healthy samples in site {new_site}: {len(none_samples)}/{len(site_data)}")
    print(f"Samples used for training the adjustment model for site {new_site}: {num_samples}/{len(site_data)}")
    print()
    
    # Prepare data for prediction
    site_data1 = site_data.copy()
    site_data1 = site_data1.loc[:, columns]
    site_data1['site'] = selected_site
    
    # Get original predictions
    predictions_df = main_model.predict_all(site_data1, train_data=main_traindata)
    
    # Prepare data for the adjustment model
    site_sub_data1 = site_sub_data.copy()
    site_sub_data1 = site_sub_data1.loc[:, columns]
    site_sub_data1['site'] = selected_site
    
    new_site_subpred = main_model.predict_all(site_sub_data1, train_data=main_traindata)
    
    # Get the available parameters from the main model
    available_params = robjects.r(f"{main_model.model_name}$parameters")
    param_names = list(available_params)

    for param in param_names:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(new_site_subpred[f'{param}_pred'])
            robjects.globalenv[f'{param}_pred_temp'] = r_data
    
    # Fit the adjustment model
    gam_refit_name = 'gam_refit'

    # Generate the R code for the refit model dynamically
    r_code = f"""
    model <- gamlss(y ~ 1 + {create_offset_term('mu', find_link(main_model.model_name, param='mu'), suffix="_pred_temp")},
    """

    for param in param_names[1:]:  # Skip 'mu' as it's already included
        r_code += f"{param}.formula = ~ 1 + {create_offset_term(param, find_link(main_model.model_name, param=param), suffix='_pred_temp')},\n"

    # Remove the trailing  newline
    r_code = r_code.rstrip('\n')

    # Add the family and method
    family = robjects.r(f"{main_model.model_name}$family")[0]
    r_code += f"""
    family = {family}(),
    method = RS({iterations})
    )
    """

    # Fit the adjustment model
    gam_refit_name = 'gam_refit'
    gamlss_model_refit = Gamlss(gam_refit_name, x_vals, y_val)
    #print(f"r_code: {r_code}")

    # Print the full R code
    #print(f"Full R code to be executed:\n{r_code}")

    gamlss_model_refit.fit(r_code=r_code, data=site_sub_data1)
    
    for param in param_names:
        robjects.r(f'rm({param}_pred_temp)')
    
    # Extract adjustment coefficients and apply adjustments
    adjusted_predictions = adjust_predictions(predictions_df, gam_refit_name)
    
    # Calculate z-scores
    z_scores = calculate_z_score_transfer(adjusted_predictions, site_data1, y_val, func_name=family)

    return predictions_df, adjusted_predictions, z_scores, train_set_index

# # Multi site transfer function
def multisite_transfer(main_model_path, main_traindata, df2, new_sites, selected_sites, fractions, y_val, x_vals, iterations=100, save_folder=None, file_name=None):
    """
    Perform site transfer for multiple new sites and optionally save the results.
    
    Parameters:
    - main_model_path: Path to the main GAMLSS model
    - main_traindata: Training dataset for the main model (pandas DataFrame)
    - df2: Test dataset (pandas DataFrame)
    - new_sites: List of new sites to transfer to
    - selected_sites: Either a single string (proxy site) or a list of strings (same length as new_sites)
    - fractions: Either a single float or a list of floats (same length as new_sites)
    - y_val: Name of the target variable (string)
    - x_vals: List of predictor variable names (list of strings)
    - iterations: Number of iterations for the adjustment model (int)
    - save_folder: Folder path to save the results (string or None)
    - file_name: File name to save the results (string or None)
    
    Returns:
    - Dictionary with results for each new site
    """
    
    # Ensure selected_sites and fractions are lists of the same length as new_sites
    if isinstance(selected_sites, str):
        selected_sites = [selected_sites] * len(new_sites)
    if isinstance(fractions, (int, float)):
        fractions = [fractions] * len(new_sites)
    
    assert len(new_sites) == len(selected_sites) == len(fractions), "Length mismatch in input parameters"
    
    results = {}
    
    for new_site, selected_site, fraction in tqdm(zip(new_sites, selected_sites, fractions)):
        # print the three zipped ones
        print(f"new_site: {new_site}, selected_site: {selected_site}, fraction: {fraction}")
        predictions_df, adjusted_predictions, z_scores, train_set_index = site_transfer(
            main_model_path, main_traindata, df2, new_site, selected_site, fraction, y_val, x_vals, iterations
        )
        
        results[new_site] = {
            "unadjusted_predictions": predictions_df,
            "adjusted_predictions": adjusted_predictions,
            "z_scores": z_scores,
            "train_set_index": train_set_index,
            "selected_site": selected_site,
            "fraction": fraction
        }
    
    # Save results if save_folder and file_name are provided
    if save_folder is not None and file_name is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        file_path = os.path.join(save_folder, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {file_path}")
    
    return results


def plot_z_scores_site(z_scores, test_data, affected=None):
        """
        Plot z-scores against age with centile lines and outlier identification.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data.
        - train_data (pandas.DataFrame): DataFrame containing the training data.
        - affected (numpy.ndarray, optional): Array of binary values indicating affected status.
        - func_name (str): Distribution type (default is SHASH)

        Returns:
        - None
        """

        s = 20
        edgecolors = 'skyblue'

        age = test_data.loc[:, 'age'].values

        # Create a scatter plot of z-scores against age
        plt.figure(figsize=(10, 6))
        plt.scatter(age, z_scores, color='blue', s=s, facecolors='none', edgecolors=edgecolors, label='Z-scores')

        # Add centile lines
        centiles = [2.5, 50, 97.5]
        colors = ['red', 'green', 'red']
        labels = ['2.5th Percentile', '50th Percentile', '97.5th Percentile']
        for centile, color, label in zip(centiles, colors, labels):
            centile_value = norm.ppf(centile / 100)
            plt.axhline(centile_value, color=color, linestyle='--', label=label)

        # Identify and highlight outliers
        outlier_indices = np.where((z_scores > 1.96) | (z_scores < -1.96))[0]
        plt.scatter(age[outlier_indices], z_scores[outlier_indices], color='red', marker='x', s=s, label='Outliers')

        # If affected status is provided, highlight affected individuals with different colors
        if affected is not None:
            unique_labels = affected.unique()
            color_map = plt.cm.get_cmap('tab10', len(unique_labels))
            for i, label in enumerate(unique_labels):
                if label != 'None':
                    label_indices = np.where(affected == label)[0]
                    color = color_map(i)
                    plt.scatter(age[label_indices], z_scores[label_indices], color=color, marker='o', s=50,
                                label=label, alpha=0.5)

        plt.xlabel('Age')
        plt.ylabel('Z-score')
        plt.title('Z-scores vs. Age')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.show()
