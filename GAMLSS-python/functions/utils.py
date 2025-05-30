import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
import logging
import shutil
import os
import numpy as np
import pandas as pd
from nilearn import datasets
from difflib import get_close_matches, SequenceMatcher
import ast
import re


gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')
utils = importr('utils')
grdevices = importr('grDevices')


def add_extra_site_data(train_data, test_data, fraction=0.2):
    """
    Add a fraction of 'None' samples from extra sites in the test dataset to the training dataset.

    Parameters:
    - train_data (pandas.DataFrame): DataFrame containing the training data.
    - test_data (pandas.DataFrame): DataFrame containing the test data.
    - fraction (float): Fraction of 'None' samples to add from extra sites (default: 0.2).

    Returns:
    - train_data (pandas.DataFrame): Updated training dataset with added samples from extra sites.
    - test_data (pandas.DataFrame): Updated test dataset with removed samples added to the training dataset.
    """
    train_sites = set(train_data['site'].unique())
    test_sites = set(test_data['site'].unique())
    extra_sites = test_sites - train_sites

    if extra_sites:
        print(f"Extra sites found in the test set: {list(extra_sites)}")
        for site in extra_sites:
            site_data = test_data[test_data['site'] == site]
            none_samples = site_data[site_data['affected'] == 'None']
            num_samples = int(len(none_samples) * fraction)
            selected_samples = none_samples.sample(n=num_samples, random_state=42)
            train_data = pd.concat([train_data, selected_samples])
            test_data = test_data.drop(selected_samples.index)

            train_data = train_data.reset_index(drop=True)  
            test_data = test_data.reset_index(drop=True)

        train_data = train_data.drop(['affected'], axis=1)

        return train_data, test_data

def check_test_sites(train_data, test_data):
    """
    Check if the test dataset has more sites than the training dataset and print the distribution of sites.

    Parameters:
    - train_data (pandas.DataFrame): DataFrame containing the training data.
    - test_data (pandas.DataFrame): DataFrame containing the test data.

    Returns:
    - None
    """
    train_sites = set(train_data['site'].unique())
    test_sites = set(test_data['site'].unique())

    print(f"Number of unique sites in the training set: {len(train_sites)}")
    print(f"Number of unique sites in the test set: {len(test_sites)}")

    extra_sites = test_sites - train_sites
    if extra_sites:
        print(f"Test sites not present in the training set: {list(extra_sites)}")
        print(f"Number of extra sites: {len(extra_sites)}")

        site_record = []
        for site in extra_sites:
            site_data = test_data[test_data['site'] == site]
            # print(f"\nDistribution for site {site}:")
            # print(f"'None': {len(site_data[site_data['affected'] == 'None'])}")
            # print("Morbidities:")
            # for morbidity in site_data['affected'].unique():
            #     if morbidity != 'None':
                    # print(f"{morbidity}: {len(site_data[site_data['affected'] == morbidity])}")

            morbidities =  list(site_data['affected'].unique())
            if 'None' in morbidities:
                morbidities.remove('None')
                
            site_data1 = site_data[site_data['affected'].isin(morbidities)]
            site_record.append({'site': site,'Length': len(site_data), 'Healthy': len(site_data[site_data['affected'] == 'None']), 'Morbidities': f"Len: {len(site_data1)}, {morbidities}"})

        site_record_df= pd.DataFrame.from_records(site_record)
        display(site_record_df)
        
        print("Suggestion: Use 20% of 'None' samples from extra sites in the training data.")
        print("This can be done by calling the 'add_extra_site_data' function.")
    else:
        print("No extra sites found in the test set.")


 
def rename_y_vals(df, old_names, new_names):
    """
    Rename y_val variables in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the results.
    old_names (list): List of current y_val names.
    new_names (list): List of new y_val names to replace the old ones.
    
    Returns:
    pd.DataFrame: DataFrame with renamed y_val variables.
    """
    if len(old_names) != len(new_names):
        raise ValueError("The number of old names must match the number of new names.")
    
    name_map = dict(zip(old_names, new_names))
    df['y_val'] = df['y_val'].map(name_map)
    
    return df




def match_regions(df, atlas=None, column='AIC_diff', region_column='y_val'):
    """
    Match regions from a DataFrame to Destrieux surface atlas regions based on similarity.

    Parameters:
    df (pd.DataFrame): DataFrame containing region names and AIC differences.
    atlas (nilearn.datasets.struct.Bunch): Atlas object. If None, uses Destrieux 2009 atlas.
    column (str): Name of the column in df containing AIC differences.
    region_column (str): Name of the column in df containing region names.

    Returns:
    tuple: (closest_match_df, left_surf_data, right_surf_data)
        closest_match_df (pd.DataFrame): DataFrame with matched regions and their scores.
        left_surf_data (np.array): Array of AIC differences for matched regions in left hemisphere.
        right_surf_data (np.array): Array of AIC differences for matched regions in right hemisphere.
    """

    def similarity_score(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Load the destrieux_2009 surface atlas
    atlas = datasets.fetch_atlas_surf_destrieux()

    # Create a dictionary mapping region names to AIC differences
    region_dict = dict(zip(df[region_column], df[column]))

    # Prepare a list to store all potential matches
    all_matches = []

    for region in atlas.labels:
        region_name = region.decode('utf-8')
        if region_name == 'Unknown':
            continue

        for hemisphere in ['L ', 'R ']:
            full_region_name = f"{hemisphere}{region_name}"
            for key in region_dict.keys():
                score = similarity_score(full_region_name, key)
                all_matches.append((full_region_name, key, score, region_dict[key]))

    # Sort all matches by similarity score in descending order
    all_matches.sort(key=lambda x: x[2], reverse=True)

    # Prepare a list to store all matches
    matches = []

    # Assign matches, ensuring each region and key is used only once
    used_regions = set()
    used_keys = set()

    for region, key, score, col_val in all_matches:
        if region not in used_regions and key not in used_keys:
            matches.append({
                'Region': region,
                'Closest_Match': key,
                'Similarity_Score': score,
                column: col_val
            })
            
            used_regions.add(region)
            used_keys.add(key)

    # Create the DataFrame from the list of matches
    closest_match_df = pd.DataFrame(matches)

    # Sort the final DataFrame by similarity score
    if not closest_match_df.empty:
        closest_match_df = closest_match_df.sort_values('Similarity_Score', ascending=False).reset_index(drop=True)

    # Create lookup arrays for left and right hemispheres
    lookup_array_left = np.zeros(len(atlas.labels))
    lookup_array_right = np.zeros(len(atlas.labels))

    for i, region in enumerate(atlas.labels):
        region_name = region.decode('utf-8')
        if region_name == 'Unknown':
            continue

        # Find the closest match in the left hemisphere
        closest_match_left = get_close_matches(f"L {region_name}", closest_match_df['Region'], n=1, cutoff=0.6)
        if closest_match_left:
            col_left = closest_match_df.loc[closest_match_df['Region'] == closest_match_left[0], column].values[0]
            lookup_array_left[i] = col_left

        # Find the closest match in the right hemisphere
        closest_match_right = get_close_matches(f"R {region_name}", closest_match_df['Region'], n=1, cutoff=0.6)
        if closest_match_right:
            col_right = closest_match_df.loc[closest_match_df['Region'] == closest_match_right[0], column].values[0]
            lookup_array_right[i] = col_right

    # Map the lookup arrays to the left and right hemispheres
    left_surf_data = lookup_array_left[atlas.map_left]
    right_surf_data = lookup_array_right[atlas.map_right]

    return closest_match_df, left_surf_data, right_surf_data


def convert_string_to_list(df, column_names):
    """
    Convert a column containing string representations of lists back into actual lists.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame
    column_name (str): The name of the column to convert
    
    Returns:
    pandas.DataFrame: A new DataFrame with the specified column converted to lists
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    for column_name in column_names:
        # Convert the string representation of lists to actual lists
        df_copy[column_name] = df_copy[column_name].apply(ast.literal_eval)
    
    return df_copy


def get_permutation_seeds(main_folder, y_val, model_type='target'):
    model_dir = os.path.join(main_folder, 'model_cache', 'saved_models')
    pattern = fr"model_{y_val}_{model_type}_perm_(\d+)\.rds$"
    
    seeds = []
    for filename in os.listdir(model_dir):
        match = re.match(pattern, filename)
        if match:
            seeds.append(int(match.group(1)))
    
    return sorted(seeds)


def remove_special_characters(dataframes):
    modified_dfs = []
    name_maps = []
    
    # Pattern to match special characters not allowed in R column names
    pattern = r'[^a-zA-Z0-9_.]'
    
    for df in dataframes:
        name_mapping = {}
        new_columns = []
        for col in df.columns:
            new_col = re.sub(pattern, '_', col)
            new_col = new_col.strip('_')  # Remove leading/trailing underscores
            name_mapping[new_col] = col
            new_columns.append(new_col)
        
        df.columns = new_columns
        modified_dfs.append(df)
        name_maps.append(name_mapping)
    
    return modified_dfs, name_maps
