import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.rinterface_lib import callbacks
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display, Image
import logging
import traceback
import sys
import pickle
import os
import re

gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')
utils = importr('utils')
grdevices = importr('grDevices')



 


class Gamlss:
    def __init__(self, model_name, x_vals, y_val):
        self.model_name = model_name
        self.x_vals = x_vals
        self.y_val = y_val

    def fit(self, r_code, data):
        """
        Fit a GAMLSS model using the provided R code.

        Parameters:
        - r_code (str): R code as a string, defining the model formula and other arguments.
        - data (pandas.DataFrame): DataFrame containing the data for fitting the model.

        Returns:
        - None
        """
        global_data_name = f"{self.model_name}_traindata"
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)
            robjects.globalenv[global_data_name] = r_data
            
            # Replace 'model' with self.model_name and 'y' with self.y_val
            r_code_formatted = re.sub(r'\bmodel\b', self.model_name, r_code)
            r_code_formatted = re.sub(r'\by\b', self.y_val, r_code_formatted)
            
            # Insert the data parameter into the gamlss function call
            r_code_parts = r_code_formatted.split("gamlss(")
            if len(r_code_parts) > 1:
                r_code_final = r_code_parts[0] + f"gamlss(data = {global_data_name}, " + r_code_parts[1]
            else:
                raise ValueError("Invalid R code: 'gamlss(' not found in the provided code.")
            
            robjects.r(r_code_final)

        robjects.r(f'rm({global_data_name})')

    def predict_all(self, test_data, train_data, transform=True, verbose=False):
        """
        Predict from the fitted GAMLSS model.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data.
        - train_data (pandas.DataFrame): DataFrame containing the training data.
        - transform (bool): Whether to apply transformations. Default is True.
        - verbose (bool): Whether to print the model summary, link functions and transformations applied. Default is False.

        Returns:
        - numpy.ndarray: Predicted values for the available parameters (mu, sigma, nu, and tau).
        """
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Response apply the transformation internally and link does not
        type_val = "response" if transform else "link"
        train_data_name = f"{self.model_name}_traindata"
        test_data_name = f"{self.model_name}_newdata"

        x_vals = self.x_vals
        y_val = self.y_val
        columns = x_vals + [y_val]

        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_train_data = robjects.conversion.py2rpy(train_data.loc[:,columns])
            r_test_data = robjects.conversion.py2rpy(test_data.loc[:,x_vals])
            robjects.globalenv[train_data_name] = r_train_data
            robjects.globalenv[test_data_name] = r_test_data

            # Update model's data reference
            robjects.r(f"""
                {self.model_name}$data <- {train_data_name}
                {self.model_name}$call$data <- as.name("{train_data_name}")
            """)

            # Suppress R output
            captured_output = io.StringIO()
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                # Get the available parameters from the fitted model
                available_params = robjects.r(f"{self.model_name}$parameters")
                param_names = list(available_params)
                link_funcs = []

                r_code = f"""
                    m_predicted <- predictAll({self.model_name}, newdata = {test_data_name}, type = "{type_val}", se.fit = FALSE)
                """

                for param in param_names:
                    link_func = robjects.r(f"{self.model_name}${param}.link")[0]
                    link_funcs.append(link_func)
                    r_code += f"""
                        {param}_pred <- m_predicted${param}
                    """

                r_code += f"""
                    data.frame({', '.join(f'{param}_pred' for param in param_names)})
                """
                
                predictions = robjects.r(r_code)

            # Print verbose information if requested
            if verbose:
                print("Model summary:")
                summary = robjects.r(f"capture.output(summary({self.model_name}))")
                for line in summary:
                    print(line)
                
                print("\nLink functions:")
                for param, link_func in zip(param_names, link_funcs):
                    print(f"{param}_link: {link_func}")

                print("\nTransformations applied:" if transform else "\nNo transformations applied.")

            # Remove the assigned variables
            robjects.r(f'rm({train_data_name})')
            robjects.r(f'rm({test_data_name})')

            return predictions
 
            
    def z_score(self, test_data, train_data, func_name='SHASH'):
        """
        Calculate Z-scores for the test data.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data.
        - train_data (pandas.DataFrame): DataFrame containing the training data.
        - func_name (str): Distribution type (default is SHASH)

        Returns:
        - numpy.ndarray: Z-scores for the test data.
        """

        func_name = 'p' + func_name
        predictions = self.predict_all(test_data, train_data, transform=True)
        param_names = predictions.columns
        
        train_data_name = f"{self.model_name}_traindata"
        test_data_name = f"{self.model_name}_newdata"

        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_train_data = robjects.conversion.py2rpy(train_data)
            r_test_data = robjects.conversion.py2rpy(test_data)
            robjects.globalenv[train_data_name] = r_train_data
            robjects.globalenv[test_data_name] = r_test_data

            for param in param_names:
                param_pred = predictions[param].values
                robjects.globalenv[f'{param}'] = robjects.FloatVector(param_pred)
            
            param_names_clean = [name.replace('_pred', '') for name in param_names]
            param_inputs = ", ".join([f"{param} = {param}_pred" for param in param_names_clean])

            r_code = f"""
            m_quantiles <- {func_name}({test_data_name}$y, {param_inputs})

            z_scores <- qnorm(m_quantiles, mean = 0, sd = 1)
            z_scores
            """
            r_code = re.sub(r'\by\b', self.y_val, r_code)
            
            # Capture and suppress the R output
            z_scores = robjects.r(r_code)

        # Remove the assigned variables
        robjects.r(f'rm({train_data_name})')
        robjects.r(f'rm({test_data_name})')
        for param in param_names:
            robjects.r(f'rm({param})')

        return np.array(z_scores)


    def plot_z_scores(self, test_data, train_data, affected=None, func_name='SHASH'):
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

        z_scores = self.z_score(test_data, train_data, func_name=func_name)
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
        # plt.scatter(age[outlier_indices], z_scores[outlier_indices], color='red', marker='x', s=s, label='Outliers')

        # If affected status is provided, highlight affected individuals with different colors
        if affected is not None:
            unique_labels = affected.unique()
            color_map = plt.cm.get_cmap('tab20', len(unique_labels))
            for i, label in enumerate(unique_labels):
                if pd.notna(label) and label != 'None':
                    label_indices = np.where(affected == label)[0]
                    color = color_map(i)
                    plt.scatter(age[label_indices], z_scores[label_indices], facecolors='none', edgecolors=color, marker='o', s=50,
                                label=label)

        plt.xlabel('Age')
        plt.ylabel('Z-score')
        plt.title('Z-scores vs. Age')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.show()



    def centiles(self, test_data, train_data, centiles=[3, 10, 25, 50, 75, 90, 97], func_name='SHASH'):
        """
        Calculate centiles for the test data.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data.
        - train_data (pandas.DataFrame): DataFrame containing the training data.
        - centiles (list): List of centiles to calculate (default is [3, 10, 25, 50, 75, 90, 97])
        - func_name (str): Distribution type (default is SHASH)

        Returns:
        - numpy.ndarray: Centile values for the test data.
        """
    
        if self.y_val not in train_data.columns:
            raise ValueError(f"The column '{self.y_val}' is not present in the training data.")

        func_name = 'q' + func_name
        predictions = self.predict_all(test_data, train_data, transform=True)
        param_names = predictions.columns
        
        train_data_name = f"{self.model_name}_traindata"
        test_data_name = f"{self.model_name}_newdata"

        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_train_data = robjects.conversion.py2rpy(train_data)
            r_test_data = robjects.conversion.py2rpy(test_data)
            robjects.globalenv[train_data_name] = r_train_data
            robjects.globalenv[test_data_name] = r_test_data

            for param in param_names:
                param_pred = predictions[param].values
                robjects.globalenv[f'{param}'] = robjects.FloatVector(param_pred)
            
            param_names_clean = [name.replace('_pred', '') for name in param_names]
            param_inputs = ", ".join([f"{param} = {param}_pred" for param in param_names_clean])

            centiles_r = robjects.FloatVector([c/100 for c in centiles])
            robjects.globalenv['centiles'] = centiles_r

            r_code = f"""
            centile_values <- sapply(centiles, function(p) {{
                {func_name}(p, {param_inputs})
            }})
            centile_values
            """
            
            centile_values = np.array(robjects.r(r_code))

        # Remove the assigned variables
        robjects.r(f'rm({train_data_name})')
        robjects.r(f'rm({test_data_name})')
        for param in param_names:
            robjects.r(f'rm({param})')

        return centile_values



    def plot_centiles(self, test_data, train_data, centiles=[3, 10, 25, 50, 75, 90, 97], func_name='SHASH', affected=None, variables_to_split=None, raw_subset=False, x_axis={'age': 'Age'},  additional_global_vars=None,plot_scatter=True):
        """
        Plot centiles against age with centile lines and optional outlier identification.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data.
        - train_data (pandas.DataFrame): DataFrame containing the training data.
        - centiles (list): List of centiles to plot (default is [3, 10, 25, 50, 75, 90, 97])
        - func_name (str): Distribution type (default is SHASH)
        - affected (numpy.ndarray, optional): Array of binary values indicating affected status.
        - variables_to_split (dict, optional): Dictionary of variables and their values to split the plot by.
        - raw_subset (bool or str): If False, use all raw data. If a column name is provided, use data from the selected category of that column.
        - x_axis (dict): Dictionary with column name as key and display name as value for x-axis.
        - plot_scatter (bool, optional): To enable or disable scatter plot in centile plot

        Returns:
        - None
        """
        # Add this check at the beginning
        if affected is not None:
            if len(test_data) != len(affected):
                raise ValueError("Length mismatch: 'affected' might be given from test data while training data is being used for plotting")

        def is_categorical(series):
            return series.dtype == 'object' or series.nunique() < 10

        def get_grid_data(test_data, fixed_values=None):
            x_col = list(x_axis.keys())[0]
            x_min, x_max = test_data[x_col].min(), test_data[x_col].max()
            x_grid = np.linspace(x_min, x_max, 200)
            
            # Initialize with a row from test_data to keep original values
            grid_data = pd.DataFrame([test_data.iloc[0].copy() for _ in range(len(x_grid))])
            
            # Set x-axis values
            grid_data[x_col] = x_grid
            
            # Update fixed values (from variables_to_split)
            if fixed_values:
                for col, val in fixed_values.items():
                    grid_data[col] = val
            
            # Update additional global variables if specified
            if additional_global_vars:
                for col in additional_global_vars:
                    if col in test_data.columns and col != x_col and col != self.y_val:
                        if is_categorical(test_data[col]):
                            grid_data[col] = test_data[col].mode().iloc[0]
                        else:
                            grid_data[col] = test_data[col].mean()
            
            return grid_data

        def plot_single(grid_data, test_data, train_data, title_suffix):
            x_col = list(x_axis.keys())[0]
            x_label = list(x_axis.values())[0]
    
            centile_values = self.centiles(grid_data, train_data, centiles, func_name)
    
            plt.figure(figsize=(8, 5))
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(centiles)))
            for i, (centile, color) in enumerate(zip(centiles, colors)):
                plt.plot(grid_data[x_col], centile_values[:, i], color=color, linestyle='-', label=f'{centile}th Centile')
    
            # Only plot scatter points if plot_scatter is True
            if plot_scatter:
                plt.scatter(test_data[x_col], test_data[self.y_val], color='blue', s=20, facecolors='none', 
                          edgecolors='skyblue', label='Actual Values')
    
                if affected is not None:
                    # This whole affected plotting section also goes inside the plot_scatter condition
                    unique_labels = pd.Series(affected).dropna().unique()
                    color_map = plt.cm.get_cmap('tab20', len(unique_labels))
                    
                    if raw_subset:
                        if raw_subset in test_data.columns:
                            selected_category = grid_data[raw_subset].iloc[0]
                            scatter_data = test_data[test_data[raw_subset] == selected_category]
                        else:
                            print(f"Warning: Column '{raw_subset}' not found in test_data. Using all raw data.")
                            scatter_data = test_data
    
                    for i, label in enumerate(unique_labels):
                        if pd.notna(label) and label != 'None':
                            label_indices = affected == label
                            color = color_map(i)
                            plt.scatter(scatter_data.loc[label_indices, x_col], scatter_data.loc[label_indices, self.y_val],
                                        facecolors='none', edgecolors=color, marker='o', s=50, label=label)

            plt.xlabel(x_label)
            plt.ylabel(self.y_val)
            plt.title(f'Centile Plot for {self.y_val} {title_suffix}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            print("\nValues used for centile calculation in this plot:")
            # Print only relevant variables
            if combination:  # Print split variables
                for var, val in combination.items():
                    print(f"{var}: {val}")
            if additional_global_vars:  # Print additional global variables
                for col in additional_global_vars:
                    if col in grid_data.columns and col != x_col and (not combination or col not in combination):
                        print(f"{col}: {grid_data[col].iloc[0]}")

        if variables_to_split is None:
            grid_data = get_grid_data(test_data)
            plot_single(grid_data, test_data, train_data, "")
        else:
            combinations = [{}]
            for var, values in variables_to_split.items():
                combinations = [dict(comb, **{var: val}) for comb in combinations for val in values]

            for combination in combinations:
                filtered_data = test_data.copy()
                title_parts = []
                for var, val in combination.items():
                    filtered_data = filtered_data[filtered_data[var] == val]
                    title_parts.append(f"{var}={val}")

                if len(filtered_data) > 0:
                    grid_data = get_grid_data(filtered_data, fixed_values=combination)
                    plot_single(grid_data, filtered_data, train_data, f"({', '.join(title_parts)})")
                else:
                    print(f"No data for combination: {combination}")



    def plot_continuous_centiles(self, test_data, train_data, centiles=[3, 10, 25, 50, 75, 90, 97], 
                    func_name='SHASH', discrete_split=None, continuous_split=None, 
                    approximated=None, x_axis={'age': 'Age'}, y_axis=None, affected=None, additional_global_vars=None):
        """
        Plot centiles against age with support for both discrete and continuous variable splitting, with uncertainty visualization.

        Parameters:
        - test_data (pandas.DataFrame): DataFrame containing the test data
        - train_data (pandas.DataFrame): DataFrame containing the training data
        - centiles (list): List of centiles to plot (default is [3, 10, 25, 50, 75, 90, 97])
        - func_name (str): Distribution type (default is SHASH)
        - discrete_split (dict, optional): Dictionary with column name as key and list of values to split by
        - continuous_split (dict, optional): Dictionary with column name as key and tuple of (display_name, n_splits) as value
        - approximated (dict, optional): Dictionary of variables with their fixed values to use instead of mean/mode
        - x_axis (dict): Dictionary with column name as key and display name as value for x-axis
        - y_axis (str, optional): Name of y-axis variable (defaults to self.y_val)
        - affected (numpy.ndarray, optional): Array of binary values indicating affected status
        - additional_global_vars (list): List of column names to include in global stats calculation

        Returns:
        - None

        """

        if y_axis is None:
            y_axis = self.y_val

        def create_ranges(data, var_name, n_splits):
            min_val = data[var_name].min()
            max_val = data[var_name].max()
            ranges = np.linspace(min_val, max_val, n_splits + 1)
            return [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]

        # First calculate or get the global values
        x_col = list(x_axis.keys())[0]
        train_columns = train_data.columns.tolist()
        
        # Modified global stats calculation
        global_stats = {}
        if approximated:
            for col, val in approximated.items():
                global_stats[col] = val
                print(f"{col}: user provided = {global_stats[col]}")
                
        if additional_global_vars:
            for col in additional_global_vars:
                if col in train_columns and col != x_col and col != self.y_val:
                    if col not in global_stats:  # Skip if already in approximated
                        if pd.api.types.is_numeric_dtype(test_data[col]):
                            global_stats[col] = test_data[col].mean()
                            print(f"{col}: mean = {global_stats[col]}")
                        else:
                            global_stats[col] = test_data[col].mode().iloc[0]
                            print(f"{col}: mode = {global_stats[col]}")


        # Filter test_data based on global_stats before any other processing
        filtered_test_data = test_data.copy()
        for col, val in global_stats.items():
            filtered_test_data = filtered_test_data[filtered_test_data[col] == val]
            # print(f"Filtered test data based on {col}={val}")
            # print(filtered_test_data.shape)
        # Add verification print statements here
        # print("\nAfter global stats filtering:")
        # print(f"Total samples: {len(filtered_test_data)}")
        # print(f"Unique sites: {filtered_test_data['site'].unique()}")

        # After filtering test_data based on global_stats
        # print("\nAfter global stats filtering:")
        # print(f"Total samples: {len(filtered_test_data)}")
        if affected is not None:
            print("Affected distribution after filtering:")
            affected = affected.fillna("None")
            print(np.unique(affected[filtered_test_data.index], return_counts=True))


        def get_grid_data(test_data, original_test_data, discrete_value=None, continuous_range=None):
            x_col = list(x_axis.keys())[0]
            
            # Store the original indices first
            original_indices = test_data.index
            
            x_min, x_max = test_data[x_col].min(), test_data[x_col].max()
            
            # Adaptive x-grid based on data distribution
            x_unique = np.sort(test_data[x_col].unique())
            x_gaps = np.diff(x_unique)
            median_gap = np.median(x_gaps)
            
            n_points = max(int((x_max - x_min) / median_gap), 50)  
            x_grid = np.linspace(x_min, x_max, min(n_points, 100))
            
            if continuous_range:
                var_name, (range_min, range_max) = continuous_range
                range_size = range_max - range_min
                
                if range_size < 1:
                    n_values = 3
                elif range_size < 10:
                    n_values = 5
                else:
                    n_values = max(7, min(int(range_size/5), 15))
                
                values = np.linspace(range_min, range_max, n_values)
                all_grid_data = []
                
                for val in values:
                    step = max(1, len(x_grid) // (n_values * 2))
                    reduced_x_grid = x_grid[::step]
                    
                    # Initialize with a row from test_data to keep original values
                    grid_data = pd.DataFrame([test_data.iloc[0].copy() for _ in range(len(reduced_x_grid))])
                    
                    # Only update the columns we need to change
                    grid_data[x_col] = reduced_x_grid
                    if discrete_value is not None and discrete_split:
                        grid_data[list(discrete_split.keys())[0]] = discrete_value
                    grid_data[var_name] = val
                    for col, val in global_stats.items():
                        grid_data[col] = val
                    
                    all_grid_data.append(grid_data)
                
                grid_data = pd.concat(all_grid_data, ignore_index=True)
            else:
                # Initialize with a row from test_data to keep original values
                grid_data = pd.DataFrame([test_data.iloc[0].copy() for _ in range(len(x_grid))])
                
                # Only update the columns we need to change
                grid_data[x_col] = x_grid
                if discrete_value is not None and discrete_split:
                    grid_data[list(discrete_split.keys())[0]] = discrete_value
                for col, val in global_stats.items():
                    grid_data[col] = val
            
            return grid_data[train_columns], original_indices

        # Now use filtered_test_data instead of test_data for all subsequent operations
        results = {}
        
        # Get discrete variable and its values
        if discrete_split:
            discrete_var, discrete_values = list(discrete_split.items())[0]
        else:
            discrete_var, discrete_values = None, [None]
                
        # Get continuous variable and number of splits
        if continuous_split:
            cont_var, (display_name, n_splits) = list(continuous_split.items())[0]
            ranges = create_ranges(filtered_test_data, cont_var, n_splits)
        else:
            cont_var, ranges = None, [None]
            display_name = None

        # Create plots for each combination
        for discrete_val in discrete_values:
            subset_data = filtered_test_data.copy()
            # print(f"subset data before {subset_data}")
            if discrete_val is not None:
                subset_data = subset_data[subset_data[discrete_var] == discrete_val]
                # print(f"subset data after {subset_data}")
                # print(f"\nFor {discrete_var}={discrete_val}:")
                key = f"{discrete_var}_{discrete_val}"
                    
            else:
                key = "all"

            
            # print(f"length o subset data: {subset_data.shape}")
            if len(subset_data) > 0:
                # print("Inside subset data > 0")
                results[key] = {}
                summed_range_data_len = 0
                # In the main function:
                for range_idx, range_tuple in enumerate(ranges):
                    if range_tuple:
                        range_min, range_max = range_tuple
                        range_data = subset_data[
                            (subset_data[cont_var] >= range_min) & 
                            (subset_data[cont_var] < range_max)
                        ]
                        
                        # print(f"\nDEBUG - Range {range_min:.2f} to {range_max:.2f}:")
                        # print(f"Range data indices: {range_data.index.tolist()[:5]}...")
                        # print(f"Number of points in range: {len(range_data)}")
                        
                        if affected is not None:
                            # print("Affected distribution in range data:")
                            range_affected = affected[range_data.index]
                            #print(np.unique(range_affected, return_counts=True))
                    else:
                        range_data = subset_data
                        range_affected = affected[range_data.index] if affected is not None else None

                    if len(range_data) > 0:
                        # print("Inside range data > 0")
                        grid_data, original_indices = get_grid_data(
                            range_data, 
                            filtered_test_data,
                            discrete_value=discrete_val,
                            continuous_range=(cont_var, (range_min, range_max)) if range_tuple else None
                        )
                        
                        # Calculate centiles using grid_data
                        # print("Before centiles")
                        # print(f"grid_data shape: {grid_data.shape}")
                        # print(f"train_data shape: {train_data.shape}")
                        # Checking if the columns of grid_data and train_data are the same and in same order

                        # Check if columns are exactly the same (including order)
                        columns_match = (grid_data.columns == train_data.columns).all()
                        # print("Grid data", grid_data.head())
                        # print("Train data", train_data.head())
                        # For more detailed comparison:
                        # print("Columns are identical and in same order:", columns_match)
                        ############################################################

                        centile_values = self.centiles(grid_data, train_data, centiles, func_name)
                        # print(f"After cetile values")
                        # Convert to numpy arrays for easier manipulation
                        x_values = grid_data[x_col].values
                        centile_values = np.array(centile_values)
                        
                        # Create bins for x values
                        n_bins = 50
                        x_bins = np.linspace(x_values.min(), x_values.max(), n_bins)
                        
                        # Initialize arrays for mean and std
                        mean_centiles = []
                        std_centiles = []
                        x_means = []
                        
                        # Group and calculate statistics
                        for i in range(len(x_bins)-1):
                            mask = (x_values >= x_bins[i]) & (x_values < x_bins[i+1])
                            if np.any(mask):
                                mean_centiles.append(np.mean(centile_values[mask], axis=0))
                                std_centiles.append(np.std(centile_values[mask], axis=0))
                                x_means.append(np.mean(x_values[mask]))
                        
                        # Convert to numpy arrays
                        mean_centiles = np.array(mean_centiles)
                        std_centiles = np.array(std_centiles)
                        x_means = np.array(x_means)
                        
                        # Store results
                        results[key][str(range_idx)] = {
                            'x_values': x_means.tolist(),
                            'centile_values': mean_centiles.tolist(),
                            'centile_std': std_centiles.tolist(),
                            'scatter_data': {
                                'x': range_data[x_col].values.tolist(),
                                'y': range_data[self.y_val].values.tolist(),
                                'indices': range_data.index.tolist()
                            }
                        }
                        # print("Inside results", results)
                        
                        if affected is not None:
                            # print("\nDEBUG - Storing affected data:")
                            # print(f"Original indices range: {min(original_indices)} to {max(original_indices)}")
                            range_affected = affected[original_indices]
                            non_none_mask = (range_affected != 'None')
                            # print(f"Number of non-None affected: {non_none_mask.sum()}")
                            
                            if non_none_mask.any():
                                results[key][str(range_idx)]['affected_data'] = {
                                    'x': range_data.loc[original_indices[non_none_mask], x_col].values.tolist(),
                                    'y': range_data.loc[original_indices[non_none_mask], self.y_val].values.tolist(),
                                    'affected': range_affected[non_none_mask].tolist(),
                                    'indices': original_indices[non_none_mask].tolist()
                                }
                                # print("Stored affected data distribution:")
                                #print(np.unique(range_affected[non_none_mask], return_counts=True))
                            
                        if range_tuple:
                            results[key][str(range_idx)]['range_info'] = {
                                'var_name': cont_var,
                                'min': range_min,
                                'max': range_max
                            }

        # Create plot
        n_rows = len(ranges)
        n_cols = len(discrete_values)
        
        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        
        # Create GridSpec for better control over spacing
        gs = fig.add_gridspec(n_rows, n_cols)
        axes = [[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)]
        axes = np.array(axes)
        
        fig.suptitle(f'Centile Plot for {y_axis}', fontsize=16)
        fig.text(0.04, 0.5, y_axis, va='center', rotation='vertical', fontsize=14)
        fig.text(0.5, 0.04, list(x_axis.values())[0], ha='center', fontsize=14)

        # Make axes 2D if needed
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot data
        first_affected_plotted = {}  # Keep track of which affected labels have been plotted

        for col_idx, discrete_val in enumerate(discrete_values):
            # print(f"Discrete value: {discrete_val}")
            key = f"{discrete_var}_{discrete_val}" if discrete_val is not None else "all"
            
            # print(f"results: {results}")
            if key in results:
                # print(f"Key: {key}")
                for row_idx in range(n_rows):
                    if str(row_idx) in results[key]:
                        ax = axes[row_idx, col_idx]
                        data = results[key][str(row_idx)]
                        
                        # Plot centiles
                        z_score = 1.96  # for 95% confidence interval
                        colors = plt.cm.coolwarm(np.linspace(0, 1, len(centiles)))
                        for i, (centile, color) in enumerate(zip(centiles, colors)):
                            # Only label centiles in first subplot
                            # print(f"Centile: {centile}")
                            # print(f"data x_values: {data['x_values']}")
                            # print(f"data centile_values: {data['centile_values']}")
                            ax.plot(data['x_values'], 
                                    np.array(data['centile_values'])[:, i], 
                                    color=color, linestyle='-', 
                                    label=f'{centile}th Centile' if row_idx == 0 and col_idx == 0 else None)
                            
                            ax.fill_between(data['x_values'],
                                        np.array(data['centile_values'])[:, i] - z_score * np.array(data['centile_std'])[:, i],
                                        np.array(data['centile_values'])[:, i] + z_score * np.array(data['centile_std'])[:, i],
                                        color=color, alpha=0.2)
                        
                        # Plot scatter points
                        ax.scatter(data['scatter_data']['x'], data['scatter_data']['y'], 
                                color='blue', s=20, facecolors='none', edgecolors='skyblue',
                                label='Actual Values' if row_idx == 0 and col_idx == 0 else None)
                        
                        # Plot affected points if provided
                        if affected is not None and 'affected_data' in data:
                            affected_data = data.get('affected_data', {})
                            
                            if affected_data:
                                x_values = affected_data.get('x', [])
                                y_values = affected_data.get('y', [])
                                affected_values = affected_data.get('affected', [])
                                
                                unique_labels = np.unique([v for v in affected_values if v != 'None'])
                                colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_labels)))
                                affected_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
                                
                                for label in unique_labels:
                                    # Only add label if we haven't plotted this affected type before
                                    should_label = label not in first_affected_plotted
                                    if should_label:
                                        first_affected_plotted[label] = True
                                    
                                    label_mask = np.array(affected_values) == label
                                    label_x = np.array(x_values)[label_mask]
                                    label_y = np.array(y_values)[label_mask]
                                    
                                    if len(label_x) > 0:
                                        ax.scatter(label_x, label_y,
                                                facecolors='none', 
                                                edgecolors=affected_colors[label], 
                                                marker='o',
                                                s=50,
                                                label=f'{label} Subjects' if should_label else None)
                        
                        # Add range information if applicable
                        if 'range_info' in data:
                            ax2 = ax.twinx()
                            ax2.set_ylabel(f"{display_name}: "
                                        f"[{data['range_info']['min']:.1f}-{data['range_info']['max']:.1f}]",
                                        rotation=270, labelpad=15)
                            ax2.set_yticks([])
                        
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                        if row_idx == 0:
                            title = f'{discrete_var}={discrete_val}' if discrete_var else ''
                            ax.set_title(title)
                        
                        ax.grid(True)

        # Gather all legend handles and labels from all subplots
        all_handles = []
        all_labels = []
        for row in range(n_rows):
            for col in range(n_cols):
                handles, labels = axes[row, col].get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in all_labels:  # avoid duplicates
                        all_handles.append(h)
                        all_labels.append(l)

        # Create single legend with all items
        fig.legend(all_handles, all_labels, bbox_to_anchor=(0.98, 0.5), loc='center left')

        # Adjust layout
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.92, top=0.95)
        plt.show()


    def save_model(self, file_path):
        """
        Save the GAMLSS model to a file using R's saveRDS function.
        
        Parameters:
        - file_path (str): Path to save the model file.
        """
        # Ensure the file has .rds extension
        if not file_path.endswith('.rds'):
            file_path += '.rds'

        r_code = f"""
        saveRDS({self.model_name}, file="{file_path}")
        """
        robjects.r(r_code)
        print(f"Model saved to {file_path}")


    @classmethod
    def load_model(cls, file_path):
        """
        Load a GAMLSS model from a file using R's readRDS function.
        
        Parameters:
        - file_path (str): Path to the saved model file.
        
        Returns:
        - Gamlss: A new Gamlss instance with the loaded model.
        """
        # Ensure the file has .rds extension
        if not file_path.endswith('.rds'):
            file_path += '.rds'

        # Load the model
        r_code = f"""
        loaded_model <- readRDS("{file_path}")
        model_call <- paste(deparse(loaded_model$call), collapse=" ")
        assign("temp_loaded_model", loaded_model)
        """
        robjects.r(r_code)

        # Extract model call
        model_call = robjects.r('model_call')[0]

        # Extract model name from the data parameter
        match = re.search(r'data\s*=\s*(\w+)_traindata', model_call)
        if match:
            model_name = match.group(1)
        else:
            model_name = "unknown_model"

        # Reassign the loaded model to the correct name in R
        robjects.r(f'{model_name} <- temp_loaded_model')
        robjects.r('rm(temp_loaded_model)')

        # Extract the formula
        formula = stats.formula(robjects.r[model_name])
        
        # Extract y_val and x_vals from the formula
        all_vars = robjects.r(f"all.vars({formula})")
        y_val = str(all_vars[0])
        x_vals = [str(var) for var in all_vars[1:]]

        # Create a new Gamlss instance
        model = cls(model_name, x_vals, y_val)
        
        # Store the loaded R model object in the Gamlss instance
        model.r_model = robjects.r[model_name]

        print(f"Model '{model_name}' loaded from {file_path}")

        return model

    ########## Used for plotting

    def get_fitted_values(self):
        model = robjects.r[self.model_name]
        fitted_values = model.rx2('mu.fv')
        return pandas2ri.rpy2py(fitted_values)

    def get_residuals(self):
        model = robjects.r[self.model_name]
        residuals = model.rx2('residuals')
        return pandas2ri.rpy2py(residuals)
    
    def get_qq_data(self):
        model = robjects.r[self.model_name]
        residuals = model.rx2('residuals')
        residuals_py = pandas2ri.rpy2py(residuals)
        n = len(residuals_py)
        theoretical_quantiles = norm.ppf(np.arange(1, n + 1) / (n + 1))
        sample_quantiles = np.sort(residuals_py)
        qq_data = pd.DataFrame({
            'theoretical_quantiles': theoretical_quantiles,
            'sample_quantiles': sample_quantiles
        })
        return qq_data
    
    def get_density_data(self):
        model = robjects.r[self.model_name]
        residuals = model.rx2('residuals')
        residuals_py = pandas2ri.rpy2py(residuals)
        kde = gaussian_kde(residuals_py, bw_method='silverman')
        quantile_residuals = np.linspace(min(residuals_py), max(residuals_py), 100)
        density = kde(quantile_residuals)
        density_data = pd.DataFrame({
            'quantile_residuals': quantile_residuals,
            'density': density
        })
        return density_data, residuals_py
    
    # Diagnostic plot
    def gamlss_diagnostic_plot(self):

        # Initialize an R graphics device
        grdevices.png(file="plot.png", width=600, height=400)
        # Plot the diagnostic plots
        base.plot(robjects.r[self.model_name])

        # Close the graphics device
        grdevices.dev_off()

        # Display the generated plot in the notebook
        display(Image(filename='plot.png'))

    def plot_diagnostics(self):
        fitted_values = self.get_fitted_values()
        residuals = self.get_residuals()
        qq_data = self.get_qq_data()
        density_data, residuals_py = self.get_density_data()
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        s = 20
        edgecolors = 'skyblue'

        # Plot against fitted values
        ax = axs[0, 0]
        ax.scatter(fitted_values, residuals, s=s, facecolors='none', edgecolors=edgecolors)
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Against Fitted Values")

        # Plot Against Index
        ax = axs[0, 1]
        ax.scatter(range(len(residuals)), residuals, s=s, facecolors='none', edgecolors=edgecolors)
        ax.set_xlabel("Index")
        ax.set_ylabel("Residuals")
        ax.set_title("Against Index")

        # Plot density estimate
        ax = axs[1, 0]
        ax.plot(density_data["quantile_residuals"], density_data["density"])
        ax.scatter(residuals_py, np.zeros_like(residuals_py), s=s, facecolors='none', edgecolors=edgecolors)
        ax.set_xlabel("Quantile Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Density Estimate")

        # Plot Q-Q plot
        ax = axs[1, 1]
        ax.scatter(qq_data["theoretical_quantiles"], qq_data["sample_quantiles"], s=s, facecolors='none', edgecolors=edgecolors)
        ax.plot(qq_data["theoretical_quantiles"], qq_data["theoretical_quantiles"], color='red', linestyle='--')
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("Normal Q-Q Plot")

        fig.suptitle("Diagnostic Plots")
        fig.tight_layout()

        plt.show()
        print(base.plot(robjects.r[self.model_name])[0][0])




######### Permutation testing

def permutation_test(m_target_func, m_base_func, data, target_var, n_permutations=1000):
    # Create a DataFrame to store the permutation results
    permutation_results = pd.DataFrame(columns=['AIC_target', 'AIC_base', 'BIC_target', 'BIC_base',
                                                'Converged_target', 'Converged_base'])
    
    # Fit the original target model
    m_target = m_target_func(data)
    
    # Fit the original base model
    m_base = m_base_func(data)
    
    # Perform permutations
    for i in range(n_permutations):
        # Permute the target variable
        permuted_target = np.random.permutation(data[target_var])
        permuted_data = data.copy()
        permuted_data[target_var] = permuted_target
        
        # Fit the target model with permuted data
        m_target_permuted = m_target_func(permuted_data)
        
        # Fit the base model with permuted data
        m_base_permuted = m_base_func(permuted_data)
        
        # Store the permutation results
        permutation_results.loc[i] = [
            stats.AIC(robjects.r[m_target_permuted.model_name])[0],
            stats.AIC(robjects.r[m_base_permuted.model_name])[0],
            stats.BIC(robjects.r[m_target_permuted.model_name])[0],
            stats.BIC(robjects.r[m_base_permuted.model_name])[0],
            robjects.r[m_target_permuted.model_name].rx2('converged')[0],
            robjects.r[m_base_permuted.model_name].rx2('converged')[0]
        ]
    
    # Calculate the real differences in AIC and BIC
    real_diff_aic = stats.AIC(robjects.r[m_target.model_name])[0] - stats.AIC(robjects.r[m_base.model_name])[0]
    real_diff_bic = stats.BIC(robjects.r[m_target.model_name])[0] - stats.BIC(robjects.r[m_base.model_name])[0]
    
    # Calculate the permutation differences in AIC and BIC
    permutation_diff_aic = permutation_results['AIC_target'] - permutation_results['AIC_base']
    permutation_diff_bic = permutation_results['BIC_target'] - permutation_results['BIC_base']
    
    # Calculate the p-values: In a non-parametric fashion
    p_value_aic = np.mean(permutation_diff_aic >= real_diff_aic)
    p_value_bic = np.mean(permutation_diff_bic >= real_diff_bic)
    
    # Calculate the convergence rates
    convergence_rate_target = np.mean(permutation_results['Converged_target'])
    convergence_rate_base = np.mean(permutation_results['Converged_base'])
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Model': ['Target', 'Base'],
        'AIC': [stats.AIC(robjects.r[m_target.model_name])[0], stats.AIC(robjects.r[m_base.model_name])[0]],
        'BIC': [stats.BIC(robjects.r[m_target.model_name])[0], stats.BIC(robjects.r[m_base.model_name])[0]],
        'P-value (AIC)': [p_value_aic, np.nan],
        'P-value (BIC)': [p_value_bic, np.nan],
        'Convergence Rate': [convergence_rate_target, convergence_rate_base]
    })
    
    return summary_df, permutation_results

def compare_models_and_save(*model_funcs, data, save_models=False, save_dir=None):
    results = []

    for model_func in model_funcs:
        model = model_func(data)
        model_name = model.model_name
        # Save models if requested
        if save_models and save_dir:
            model.save_model(os.path.join(save_dir, f"{model_name}.rds"))
            logging.info(f"Model name: {model_name} has been saved at {save_dir}")

        logging.debug(f"Model name: {model_name}")
        logging.debug(f"Available objects in R: {list(robjects.r.ls())}")

        if model_name not in robjects.r.ls():
            logging.error(f"Model '{model_name}' not found in R environment.")
            aic = None
            bic = None
            converged = None
        else:
            try:
                logging.debug(f"Accessing AIC for model: {model_name}")
                aic = stats.AIC(robjects.r[model_name])[0]
                logging.debug(f"Accessing BIC for model: {model_name}")
                bic = stats.BIC(robjects.r[model_name])[0]
                logging.debug(f"Accessing converged status for model: {model_name}")
                converged = robjects.r[model_name].rx2('converged')[0]
            except Exception as e:
                logging.error(f"Error occurred while accessing model attributes for model: {model_name}")
                logging.error(str(e))
                aic = None
                bic = None
                converged = None

        results.append({
            'Model': model_name,
            'AIC': aic,
            'BIC': bic,
            'Converged': converged
        })

    results_df = pd.DataFrame(results)
    results_df.set_index('Model', inplace=True)

    try:
        best_aic_model = results_df['AIC'].idxmin()
        best_bic_model = results_df['BIC'].idxmin()

        aic_diff = results_df['AIC'].nsmallest(2).diff().iloc[1]
        bic_diff = results_df['BIC'].nsmallest(2).diff().iloc[1]
    except (IndexError, KeyError, ValueError) as e:
        logging.error("Error occurred while determining best models and differences.")
        logging.error(str(e))
        best_aic_model = None
        best_bic_model = None
        aic_diff = None
        bic_diff = None

    summary = {
        'Best_Model_AIC': best_aic_model,
        'AIC_Diff': aic_diff,
        'Best_Model_BIC': best_bic_model,
        'BIC_Diff': bic_diff
    }

    return results_df, summary







