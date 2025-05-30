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

gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')
utils = importr('utils')
grdevices = importr('grDevices')

def keep_logs(log_folder, file_name, save_log=True):
    # Configure logging to output to both the Jupyter notebook and a log file
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all log levels

    # Clear any existing handlers (useful if running this cell multiple times)
    logger.handlers = []

    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the stream handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)

    if save_log:
        log_file = os.path.join(log_folder, file_name)
        # Check if the path exists
        if os.path.exists(log_file):
            if os.path.isfile(log_file):
                os.remove(log_file)  # Remove the file if it exists
            elif os.path.isdir(log_file):
                shutil.rmtree(log_file)  # Remove the directory and its contents if it exists
        else:
            # If the file doesn't exist, we don't need to do anything
            pass

        # Create the necessary directories for the log file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

def run_models(create_model_func, y_val, df_train, additional_features, base_formula, target_formula, 
               x_vals=['age', 'sex', 'site'], save_models=False, save_dir=None):
    """
    Run and compare GAMLSS models for a given outcome variable.

    Parameters:
    - y_val (str): The name of the outcome variable.
    - df_train (pandas.DataFrame): The training dataset.
    - additional_features (str or list): Additional feature(s) to include in the target model.
    - base_formula (dict): Formula specifications for the base model.
    - target_formula (dict): Formula specifications for the target model.
    - x_vals (list): List of predictor variables (default is ['age', 'sex', 'site']).
    - file_name (str): Name of the log file.

    Returns:
    - dict: A dictionary containing the results, including:
        - 'y_val': The outcome variable name.
        - 'results_df': DataFrame with model comparison results.
        - 'summary': Summary of the model comparison.
    """
    
    logging.info(f"Running models for y_val: {y_val}")

    # Ensure additional_features is a list
    if isinstance(additional_features, str):
        additional_features = [additional_features]

    # Add additional features to x_vals and create a subset of the training data
    x_vals = x_vals + additional_features
    columns = x_vals.copy()
    columns.extend([y_val])
    df_subset = df_train.loc[:, columns]

    # Add additional features to R global environment
    for i, feature_name in enumerate(additional_features):
        robjects.globalenv[str(feature_name)] = feature_name

    # Create base model function
    model_base_func = create_model_func(
        model_name=f'model_{y_val}_base',
        x_vals=x_vals,
        y_val=y_val,
        mu_formula=base_formula['mu_formula'],
        sigma_formula=base_formula['sigma_formula'],
        nu_formula=base_formula['nu_formula'],
        tau_formula=base_formula['tau_formula']
    )

    # Create target model function
    model_target_func = create_model_func(
        model_name=f'model_{y_val}_target',
        x_vals=x_vals,
        y_val=y_val,
        mu_formula=target_formula['mu_formula'],
        sigma_formula=target_formula['sigma_formula'],
        nu_formula=target_formula['nu_formula'],
        tau_formula=target_formula['tau_formula']
    )

    # Compare the models and save if requested
    results_df, summary = compare_models_and_save(model_base_func, model_target_func, 
                                                                 data=df_subset, 
                                                                 save_models=save_models, 
                                                                 save_dir=save_dir)

    # Clean up: remove additional features from R environment
    for i, feature_name in enumerate(additional_features):
        robjects.r(f"rm({str(feature_name)})")

    logging.info(f"Finished models for y_val: {y_val}")

    # Return results
    return {
        'y_val': y_val,
        'results_df': results_df,
        'summary': summary
    }


# Reason to define this function separately is because we are using multiprocessing and we need to make sure that the function is
# pikcleable. The issue comes from our main function "model_func" being nested inside "create_model_func" which make it not pickleable as
# "Nested functions are not picklable by default"
def model_comparison_wrapper(args):
    y_val, df_train, additional_features, base_formula, target_formula, x_vals, create_model_func, cache_dir, force_overwrite, save_models = args
    
    cache_file = os.path.join(cache_dir, f"{y_val}_result.pkl")
    save_dir = os.path.join(cache_dir, 'saved_models')
    
    if os.path.exists(cache_file) and not force_overwrite:
        logging.info(f"Loading cached result for {y_val}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        if save_models:
            os.makedirs(save_dir, exist_ok=True)        
        result = run_models(create_model_func, y_val, df_train, additional_features, base_formula, target_formula, 
                            x_vals, save_models, save_dir)
        
        os.makedirs(cache_dir, exist_ok=True)

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        logging.info(f"Cached result for {y_val}")
        return result


def model_comparison(create_model_func, y_val_columns, df_train, additional_features, base_formula, target_formula, 
                        x_vals=['age', 'sex', 'site'], main_folder=None, save_log=True, 
                        num_cores=None, force_overwrite=False, save_models=False, use_multiprocessing=True):
    """
    Run GAMLSS models in parallel or sequentially for multiple outcome variables and compile results.

    Parameters:
    - `create_model_func`: Function to create a GAMLSS model
    - `y_val_columns`: List of y-value column names to model
    - `df_train`: Training dataframe
    - `additional_features`: Additional features to include in the target model
    - `base_formula`: Dictionary specifying the base model formula
    - `target_formula`: Dictionary specifying the target model formula
    - `x_vals`: List of predictor variables (default: ['age', 'sex', 'site'])
    - `main_folder`: Folder for saving results and logs (default: 'results')
    - `save_log`: Whether to save logs (default: True)
    - `num_cores`: Number of CPU cores to use for multiprocessing (default: None, uses all available cores)
    - `force_overwrite`: Whether to overwrite existing cached results (default: False)
    - `save_models`: Whether to save fitted models (default: False)
    - `use_multiprocessing`: Whether to use multiprocessing (default: True)


    Returns:
    - tuple: (comparison_df, summary_df)
        comparison_df (pd.DataFrame): Comparison of base and target models.
        summary_df (pd.DataFrame): Summary of best models and their differences.
    """

    # Set up folders and logging
    if main_folder is None:
        main_folder = 'results'

    # Creating log and cache directories if does not exist
    cache_dir = os.path.join(main_folder, 'model_cache')
    log_folder = os.path.join(main_folder, 'model_logs')

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    keep_logs(log_folder, 'log_file.log', save_log)


    if num_cores is None:
        num_cores = 4
        
    logging.info(f"Using {'multiprocessing' if use_multiprocessing else 'for loop'} for processing")
    if use_multiprocessing:
        logging.info(f"Using {num_cores} cores for multiprocessing")

    # Check which y_vals are already processed
    processed_y_vals = [y_val for y_val in y_val_columns 
                        if os.path.exists(os.path.join(cache_dir, f"{y_val}_result.pkl")) and not force_overwrite]
    
    to_process_y_vals = [y_val for y_val in y_val_columns if y_val not in processed_y_vals]

    logging.info(f"Skipping {len(processed_y_vals)} already processed y_vals")
    logging.info(f"Processing {len(to_process_y_vals)} y_vals")

    if use_multiprocessing:
        with mp.Pool(processes=num_cores) as pool:
            args = [(y_val, df_train, additional_features, base_formula, target_formula, x_vals, create_model_func, cache_dir, force_overwrite, save_models) for y_val in to_process_y_vals]
            new_results = pool.map(model_comparison_wrapper, args)
    else:
        new_results = []
        for y_val in tqdm(to_process_y_vals, desc="Processing y_vals"):
            result = model_comparison_wrapper((y_val, df_train, additional_features, base_formula, target_formula, x_vals, create_model_func, cache_dir, force_overwrite, save_models))
            new_results.append(result)

    # Load cached results
    cached_results = []
    for y_val in processed_y_vals:
        with open(os.path.join(cache_dir, f"{y_val}_result.pkl"), 'rb') as f:
            cached_results.append(pickle.load(f))

    results = cached_results + new_results

    comparison_results = []
    summary_results = []

    for result in results:
        y_val = result['y_val']
        results_df = result['results_df']
        summary = result['summary']

        base_model = f'model_{y_val}_base'
        target_model = f'model_{y_val}_target'

        comparison_row = [y_val, results_df.loc[base_model, 'AIC'], results_df.loc[target_model, 'AIC'],
                          results_df.loc[base_model, 'BIC'], results_df.loc[target_model, 'BIC'],
                          results_df.loc[base_model, 'Converged'], results_df.loc[target_model, 'Converged']]
        comparison_results.append(comparison_row)

        summary_row = [y_val, summary['Best_Model_AIC'], summary['AIC_Diff'],
                       summary['Best_Model_BIC'], summary['BIC_Diff']]
        summary_results.append(summary_row)

    comparison_df = pd.DataFrame(comparison_results, columns=['y_val', 'base_AIC', 'target_AIC',
                                                              'base_BIC', 'target_BIC',
                                                              'base_Converged', 'target_Converged'])

    comparison_df['AIC_diff'] = comparison_df['base_AIC'] - comparison_df['target_AIC']
    comparison_df['BIC_diff'] = comparison_df['base_BIC'] - comparison_df['target_BIC']

                                            
    summary_df = pd.DataFrame(summary_results, columns=['y_val', 'Best_Model_AIC', 'AIC_Diff',
                                                        'Best_Model_BIC', 'BIC_Diff'])

    comparison_df.to_csv(os.path.join(main_folder, 'comparison_df.csv'), index=False)
    summary_df.to_csv(os.path.join(main_folder, 'summary_df.csv'), index=False)



    return comparison_df, summary_df





##################### Code for comparing multiple formulas
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import os
import pickle
import logging
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

stats = importr('stats')

def run_single_y_val(args):
    y_val, df_train, formulas, x_vals, create_model_func, cache_dir, force_overwrite, save_models, save_dir, formula_names = args
    cache_file = os.path.join(cache_dir, f"{y_val}_result.pkl")
    if os.path.exists(cache_file) and not force_overwrite:
        logging.info(f"Loading cached result for {y_val}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logging.info(f"Processing {y_val}")
    results = []
    for i, (formula, formula_name) in enumerate(zip(formulas, formula_names), 1):
        model_name = f"model_{y_val}_{formula_name}"
        model_func = create_model_func(
            model_name=model_name,
            x_vals=x_vals,
            y_val=y_val,
            mu_formula=formula['mu_formula'],
            sigma_formula=formula['sigma_formula'],
            nu_formula=formula['nu_formula'],
            tau_formula=formula['tau_formula']
        )
        model = model_func(df_train)
        aic = stats.AIC(robjects.r[model.model_name])[0]
        bic = stats.BIC(robjects.r[model.model_name])[0]
        converged = robjects.r[model.model_name].rx2('converged')[0]
        results.append({
            'formula': formula_name,
            'AIC': aic,
            'BIC': bic,
            'converged': converged
        })
        
        if save_models:
            model.save_model(os.path.join(save_dir, f"{model_name}.rds"))
    
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_multiple_formula_models(create_model_func, y_val_columns, df_train, formulas, 
                                x_vals=['age', 'sex', 'site'], main_folder=None, save_log=True, 
                                num_cores=None, force_overwrite=False, save_models=False, use_multiprocessing=True,
                                formula_names=None): 
    """
    The `run_multiple_formula_models` function allows you to compare multiple GAMLSS model formulas across different y-values. It supports parallel processing for efficiency.

    #### Function Parameters:

    - `create_model_func`: Function to create a GAMLSS model
    - `y_val_columns`: List of y-value column names to model
    - `df_train`: Training dataframe
    - `formulas`: List of formula dictionaries (each containing 'mu_formula', 'sigma_formula', 'nu_formula', 'tau_formula')
    - `x_vals`: List of predictor variables (default: ['age', 'sex', 'site'])
    - `main_folder`: Folder for saving results and logs (default: 'results')
    - `save_log`: Whether to save logs (default: True)
    - `num_cores`: Number of CPU cores to use for multiprocessing (default: None, uses all available cores)
    - `force_overwrite`: Whether to overwrite existing cached results (default: False)
    - `save_models`: Whether to save fitted models (default: False)
    - `use_multiprocessing`: Whether to use multiprocessing (default: True)
    - `formula_names`: List of names for the formulas (default: None, uses "formula1", "formula2", etc.)
    """


    # Set up folders and logging
    if main_folder is None:
        main_folder = 'results'
    # Creating log and cache directories if does not exist
    cache_dir = os.path.join(main_folder, 'model_cache')
    log_folder = os.path.join(main_folder, 'model_logs')

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    save_dir = os.path.join(cache_dir, 'saved_models')
    if save_models:
        os.makedirs(save_dir, exist_ok=True)      


    if num_cores is None:
        num_cores = 4
        
    if save_log:
        keep_logs(log_folder, 'log_file.log', save_log)


    if num_cores is None:
        num_cores = mp.cpu_count()

    logging.info(f"Using {'multiprocessing' if use_multiprocessing else 'for loop'} for processing")
    if use_multiprocessing:
        logging.info(f"Using {num_cores} cores for multiprocessing")

    if formula_names is None:
        formula_names = [f"formula{i}" for i in range(1, len(formulas) + 1)]
    
    args_list = [(y_val, df_train, formulas, x_vals, create_model_func, cache_dir, force_overwrite, save_models, save_dir, formula_names) 
                 for y_val in y_val_columns]

    if use_multiprocessing:
        with mp.Pool(processes=num_cores) as pool:
            all_results = list(tqdm(pool.imap(run_single_y_val, args_list), total=len(y_val_columns)))
    else:
        all_results = [run_single_y_val(args) for args in tqdm(args_list)]

    # Prepare comparison DataFrame
    comparison_data = []
    convergence_data = []
    summary_data = []

    for y_val, results in zip(y_val_columns, all_results):
        row = {'y_val': y_val}
        conv_row = {'y_val': y_val}
        for result in results:
            formula = result['formula']
            row[f"{formula}_AIC"] = result['AIC']
            row[f"{formula}_BIC"] = result['BIC']
            conv_row[f"{formula}_convergence"] = result['converged']
        
        best_aic = min(results, key=lambda x: x['AIC'])
        best_bic = min(results, key=lambda x: x['BIC'])
        
        row['best_formula_AIC'] = f"{best_aic['formula']} ({best_aic['AIC']:.2f})"
        row['best_formula_BIC'] = f"{best_bic['formula']} ({best_bic['BIC']:.2f})"
        
        comparison_data.append(row)
        convergence_data.append(conv_row)
        summary_data.append({
            'y_val': y_val,
            'best_AIC_formula': best_aic['formula'],
            'best_AIC_value': best_aic['AIC'],
            'best_BIC_formula': best_bic['formula'],
            'best_BIC_value': best_bic['BIC']
        })

    comparison_df = pd.DataFrame(comparison_data)
    convergence_df = pd.DataFrame(convergence_data)
    summary_df = pd.DataFrame(summary_data)

    # Save results
    comparison_df.to_csv(os.path.join(main_folder, 'comparison_df.csv'), index=False)
    convergence_df.to_csv(os.path.join(main_folder, 'convergence_df.csv'), index=False)
    summary_df.to_csv(os.path.join(main_folder, 'summary_df.csv'), index=False)

    return comparison_df, convergence_df, summary_df




############ Permutation Code ####################
import traceback
import time


def run_single_permutation(args):
    create_model_func, y_val, permuted_data, x_vals_extended, base_formula, target_formula, seed, save_models, save_dir = args
    
    
    start_time = time.time()
    logging.info(f"Starting permutation for {y_val} with seed {seed}")
    
    try:
        # Create and fit base model
        logging.info(f"Creating base model for {y_val} with seed {seed}")
        model_base_func = create_model_func(
            model_name=f'model_{y_val}_base_perm_{seed}',
            x_vals=x_vals_extended,
            y_val=y_val,
            mu_formula=base_formula['mu_formula'],
            sigma_formula=base_formula['sigma_formula'],
            nu_formula=base_formula['nu_formula'],
            tau_formula=base_formula['tau_formula']
        )

        logging.info(f"Fitting base model for {y_val} with seed {seed}")
        m_base = model_base_func(permuted_data)
        logging.info(f"Base model fitted successfully for {y_val} with seed {seed}")

        # Create and fit target model
        logging.info(f"Creating target model for {y_val} with seed {seed}")
        model_target_func = create_model_func(
            model_name=f'model_{y_val}_target_perm_{seed}',
            x_vals=x_vals_extended,
            y_val=y_val,
            mu_formula=target_formula['mu_formula'],
            sigma_formula=target_formula['sigma_formula'],
            nu_formula=target_formula['nu_formula'],
            tau_formula=target_formula['tau_formula']
        )

        logging.info(f"Fitting target model for {y_val} with seed {seed}")
        m_target = model_target_func(permuted_data)
        logging.info(f"Target model fitted successfully for {y_val} with seed {seed}")

        # Calculate AIC, BIC, and check convergence
        logging.info(f"Calculating metrics for {y_val} with seed {seed}")
        result = {
            'seed': seed,
            'AIC_target': stats.AIC(robjects.r[m_target.model_name])[0],
            'AIC_base': stats.AIC(robjects.r[m_base.model_name])[0],
            'BIC_target': stats.BIC(robjects.r[m_target.model_name])[0],
            'BIC_base': stats.BIC(robjects.r[m_base.model_name])[0],
            'Converged_target': robjects.r[m_target.model_name].rx2('converged')[0],
            'Converged_base': robjects.r[m_base.model_name].rx2('converged')[0]
        }
        
        if save_models and save_dir:
            m_base.save_model(os.path.join(save_dir, f"{m_base.model_name}.rds"))
            m_target.save_model(os.path.join(save_dir, f"{m_target.model_name}.rds"))


        end_time = time.time()
        logging.info(f"Permutation completed successfully for {y_val} with seed {seed}. Time taken: {end_time - start_time:.2f} seconds")
        return result

    except Exception as e:
        end_time = time.time()
        logging.error(f"Error in permutation for {y_val} with seed {seed}. Time taken: {end_time - start_time:.2f} seconds")
        logging.error(f"Error details: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None



def run_permutation_test(create_model_func, y_val, df_train, additional_features, base_formula, target_formula, 
                         x_vals, n_permutations, num_cores, permutation_seeds, use_multiprocessing=True, save_models=False, save_dir=None):
    
    
    logging.info(f"Starting permutation test for {y_val} with {n_permutations} permutations")

    # Ensure additional_features is a list
    if isinstance(additional_features, str):
        additional_features = [additional_features]

    x_vals_extended = x_vals + additional_features

    # Fit unshuffled models
    model_base_unshuffled = create_model_func(
        model_name=f'model_{y_val}_base_unshuffled',
        x_vals=x_vals_extended,
        y_val=y_val,
        mu_formula=base_formula['mu_formula'],
        sigma_formula=base_formula['sigma_formula'],
        nu_formula=base_formula['nu_formula'],
        tau_formula=base_formula['tau_formula']
    )

    model_target_unshuffled = create_model_func(
        model_name=f'model_{y_val}_target_unshuffled',
        x_vals=x_vals_extended,
        y_val=y_val,
        mu_formula=target_formula['mu_formula'],
        sigma_formula=target_formula['sigma_formula'],
        nu_formula=target_formula['nu_formula'],
        tau_formula=target_formula['tau_formula']
    )

    m_base_unshuffled = model_base_unshuffled(df_train)
    m_target_unshuffled = model_target_unshuffled(df_train)

    if save_models and save_dir:
        m_base_unshuffled.save_model(os.path.join(save_dir, f"{m_base_unshuffled.model_name}.rds"))
        m_target_unshuffled.save_model(os.path.join(save_dir, f"{m_target_unshuffled.model_name}.rds"))


    aic_target_unshuffled = stats.AIC(robjects.r[m_target_unshuffled.model_name])[0]
    aic_base_unshuffled = stats.AIC(robjects.r[m_base_unshuffled.model_name])[0]
    bic_target_unshuffled = stats.BIC(robjects.r[m_target_unshuffled.model_name])[0]
    bic_base_unshuffled = stats.BIC(robjects.r[m_base_unshuffled.model_name])[0]
    aic_diff_unshuffled = aic_base_unshuffled - aic_target_unshuffled
    bic_diff_unshuffled = bic_base_unshuffled - bic_target_unshuffled

    # Prepare permuted datasets
    permuted_datasets = []
    for seed in permutation_seeds[:n_permutations]:
        np.random.seed(seed)
        permuted_data = df_train.copy()
        permuted_data[y_val] = np.random.permutation(df_train[y_val])
        permuted_datasets.append(permuted_data)

    # Run permutations
    if use_multiprocessing:
        with mp.Pool(processes=num_cores) as pool:
            args_list = [(create_model_func, y_val, data, x_vals_extended, base_formula, target_formula, seed, save_models, save_dir) 
                         for data, seed in zip(permuted_datasets, permutation_seeds[:n_permutations])]
            results = pool.map(run_single_permutation, args_list)
    else:
        results = []
        for data, seed in zip(permuted_datasets, permutation_seeds[:n_permutations]):
            args = (create_model_func, y_val, data, x_vals_extended, base_formula, target_formula, seed, save_models, save_dir)
            results.append(run_single_permutation(args))

    # Filter out None results (failed permutations)
    valid_results = [result for result in results if result is not None]

    # Process results
    if valid_results:
        permutation_df = pd.DataFrame(valid_results)
        permutation_diff_aic = permutation_df['AIC_base'] - permutation_df['AIC_target']
        permutation_diff_bic = permutation_df['BIC_base'] - permutation_df['BIC_target']
        
        p_value_aic = np.mean(permutation_diff_aic >= aic_diff_unshuffled)
        p_value_bic = np.mean(permutation_diff_bic >= bic_diff_unshuffled)
        
        convergence_rate_target = np.mean(permutation_df['Converged_target'])
        convergence_rate_base = np.mean(permutation_df['Converged_base'])
        
        perm_result = {
            'y_val': y_val,
            'Unshuffled AIC_target': aic_target_unshuffled,
            'Unshuffled AIC_base': aic_base_unshuffled,
            'Unshuffled AIC_diff': aic_diff_unshuffled,
            'Unshuffled BIC_diff': bic_diff_unshuffled,
            'Permuted AIC_diff': list(permutation_diff_aic),
            'Permuted BIC_diff': list(permutation_diff_bic),
            'P-value (AIC)': p_value_aic,
            'P-value (BIC)': p_value_bic,
            'Convergence Rate': [convergence_rate_target, convergence_rate_base]
        }
    else:
        perm_result = None
        logging.warning(f"No valid permutations for {y_val}")

    # Clean up R environment
    for feature_name in additional_features:
        robjects.r(f"rm({str(feature_name)})")

    logging.info(f"Permutation test completed for {y_val}. Successful permutations: {len(valid_results)}/{n_permutations}")

    return perm_result


    
def run_models_with_permutation(create_model_func, y_val_columns, df_train, additional_features, base_formula, target_formula, 
                                x_vals=['age', 'sex', 'site'], main_folder=None, save_log=True, 
                                num_cores=None, force_overwrite=False, n_permutations=None, random_seed=None,
                                use_multiprocessing=True, save_models=False):

    # Set up folders and logging
    if main_folder is None:
        main_folder = 'results'
    # Creating log and cache directories if does not exist
    cache_dir = os.path.join(main_folder, 'model_cache')
    log_folder = os.path.join(main_folder, 'model_logs')

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    save_dir = os.path.join(cache_dir, 'saved_models')
    if save_models:
        os.makedirs(save_dir, exist_ok=True)      


    if save_log:
        keep_logs(log_folder, 'log_file.log', save_log)
    
    logging.info(f"Main folder: {main_folder}")
    logging.info(f"Cache folder: {cache_dir}")
    logging.info(f"Log folder: {log_folder}")
    
    if num_cores is None:
        num_cores = mp.cpu_count()
    logging.info(f"Using {num_cores} CPU cores")
    
    # Generate seeds for permutations
    if random_seed is None:
        random_seed = 0
    np.random.seed(random_seed)
    logging.info(f"Using random seed: {random_seed}")
    
    if n_permutations is not None:
        permutation_seeds = np.random.choice(np.arange(0, 100000000), size=10, replace=False)
        logging.info(f"Generated {n_permutations} permutation seeds")
    else:
        permutation_seeds = None
        logging.info("No permutations requested")
    
    permutation_results = {}
    
    for y_val in tqdm(y_val_columns, desc="Processing y_val columns"):
        cache_file = os.path.join(cache_dir, f"{y_val}_perm_result.pkl")
        
        if os.path.exists(cache_file) and not force_overwrite:
            logging.info(f"Loading cached permutation result for {y_val}")
            with open(cache_file, 'rb') as f:
                perm_result = pickle.load(f)
        else:
            logging.info(f"Processing permutation for {y_val}")
            
            # Run permutation test
            if n_permutations is not None:
                perm_result = run_permutation_test(create_model_func, y_val, df_train, additional_features, 
                                                base_formula, target_formula, x_vals, 
                                                n_permutations, num_cores, permutation_seeds,
                                                use_multiprocessing=use_multiprocessing,
                                                save_models=save_models,
                                                save_dir=save_dir)
                
                # Cache the permutation results
                with open(cache_file, 'wb') as f:
                    pickle.dump(perm_result, f)
                
                logging.info(f"Cached permutation results for {y_val}")
            else:
                perm_result = None
        
        if perm_result is not None:
            permutation_results[y_val] = perm_result

    # Save results
    if permutation_results:
        permutation_df = pd.DataFrame(permutation_results).T
        # Ensure y_val is the first column
        cols = permutation_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('y_val')))
        permutation_df = permutation_df[cols]
        permutation_df = permutation_df.reset_index(drop=True)
        permutation_df.to_csv(os.path.join(main_folder, 'permutation_results.csv'), index=False)
        logging.info(f"Saved permutation results to {os.path.join(main_folder, 'permutation_results.csv')}")

    return permutation_df if permutation_results else None

