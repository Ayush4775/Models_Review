import numpy as np
import pandas as pd
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def generate_synthetic_data(dist_type="normal", n_samples=2000, seed=42, **kwargs):
    """
    Generates synthetic (Age, Thickness) data where:
      - Age is uniform in [age_min, age_max].
      - Thickness is drawn from one of several possible distributions,
        always clipped to [0,1].

    Parameters
    ----------
    dist_type : str
        Which distribution to sample thickness from.
        One of ["normal", "gamma", "beta", "gmm", "long_tailed"].
    n_samples : int
        Number of samples.
    seed : int
        Random seed for reproducibility.
    kwargs : dict
        Distribution-specific parameters.
        (e.g., mean_thickness, var_thickness for "normal",
               shape_thk, scale_thk for "gamma",
               alpha_thk, beta_thk for "beta",
               means_thk, stds_thk, weights_thk for "gmm",
               df_thk, loc_thk, scale_thk for "long_tailed")

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1)
        The "Age" feature array (uniform).
    y : np.ndarray, shape (n_samples,)
        The "Cortical Thickness" target values, in [0,1].
    """
    np.random.seed(seed)

    ages = np.random.normal(loc=45, scale=25, size=n_samples)
    ages = np.clip(ages, 0, 80)  # Ensure ages are between 0 and 80
    ages = ages.astype(int)  # Convert ages to integer

    # 2. Generate Thickness according to dist_type
    if dist_type == "normal":
        # Cortical thickness mean depends linearly on age
        mean_thickness = 0.8 - (ages - 20) * (0.8 - 0.3) / (80 - 20)
        var_thickness = kwargs.get("var_thickness", 0.05)  # Reduced default variance
        std_thickness = np.sqrt(var_thickness)

        thickness_raw = np.random.normal(loc=mean_thickness, scale=std_thickness)

    elif dist_type == "gamma":
        # shape_thk=2.0, scale_thk=0.5, for example
        shape_thk = kwargs.get("shape_thk", 2.0)
        scale_thk = kwargs.get("scale_thk", 0.5)

        thickness_raw = np.random.gamma(shape_thk, scale_thk, size=n_samples)

    elif dist_type == "beta":
        # alpha_thk=2.0, beta_thk depends on age linearly
        alpha_thk = kwargs.get("alpha_thk", 2.0)
        beta_thk = 3.6 + (ages) * (42 - 3.6) / (80)  # Linear dependency on age

        thickness_raw = np.random.beta(alpha_thk, beta_thk, size=n_samples)
    elif dist_type == "gmm":
        # means_thk=[0.3, 0.7], stds_thk=[0.1, 0.05], weights_thk=[0.5, 0.5]
        # Linearly dependent means_thk based on ages
        means_thk = [
            0.8 - (ages) * (0.8 - 0.2) / (80),
            0.9 - (ages) * (0.9 - 0.3) / (80),
        ]
        stds_thk = kwargs.get("stds_thk", [0.1, 0.05])
        weights_thk = kwargs.get("weights_thk", [0.5, 0.5])

        thickness_list = []
        for _ in range(n_samples):
            comp = np.random.choice(len(means_thk), p=weights_thk)
            sample = np.random.normal(means_thk[comp][_], stds_thk[comp])
            thickness_list.append(sample)
        thickness_raw = np.array(thickness_list)

    elif dist_type == "long_tailed":
        # df_thk=3, loc_thk=0.5, scale_thk=0.2, for example
        alpha_thk = kwargs.get("alpha_thk", 1.1)
        beta_thk = kwargs.get("beta_thk", 11)

        thickness_raw = np.random.beta(alpha_thk, beta_thk, size=n_samples)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    # Convert Age to 2D shape (n_samples, 1) for scikit-learn
    X = ages.reshape(-1, 1)

    return X, thickness_raw


# def run_gpr(X, y, n_restarts=5, dist_type=""):
#     """
#     Fits a Gaussian Process Regressor on (X, y), then predicts and plots.
#     """
#     # Define kernel
#     kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
#     gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts)
#     gpr.fit(X, y)

#     # Predictions: sample a fine range of ages
#     age_min, age_max = X.min(), X.max()
#     X_test = np.linspace(age_min, age_max, 200).reshape(-1, 1)
#     y_mean, y_std = gpr.predict(X_test, return_std=True)

#     # Print out the optimized kernel info
#     print("Optimized Kernel:", gpr.kernel_)
#     print("Log Marginal Likelihood:", gpr.log_marginal_likelihood(gpr.kernel_.theta))

#     # Plot results
#     plt.figure()
#     plt.scatter(X, y, label="Samples", marker="x")
#     plt.plot(X_test, y_mean, label="GPR Mean Prediction")
#     plt.fill_between(
#         X_test.ravel(),
#         y_mean - 1.96 * y_std,
#         y_mean + 1.96 * y_std,
#         alpha=0.2,
#         label="95% Confidence Interval",
#     )
#     plt.xlabel("Age (years) (Uniform)")
#     plt.ylabel("Cortical Thickness [0,1]")
#     plt.title(f"GPR on Synthetic Data (Thickness ~ {dist_type.upper()})")
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    # Example usage:

    # 1. Select distribution for cortical thickness
    dist_type = "long_tailed"  # could be "gamma", "beta", "gmm", or "long_tailed"

    # 2. Provide parameters for that distribution
    # Normal example: mean_thickness=0.5, var_thickness=0.05
    params = {
        "age_min": 20,
        "age_max": 80,
        "mean_thickness": 0.5,
        "var_thickness": 0.05,
    }

    # For "gamma": shape_thk=2.0, scale_thk=0.5
    # For "beta": alpha_thk=2.0, beta_thk=5.0
    # For "gmm": means_thk=[0.3,0.7], stds_thk=[0.1,0.05], weights_thk=[0.5,0.5]
    # For "long_tailed": df_thk=3, loc_thk=0.5, scale_thk=0.2

    # 3. Generate data
    X, y = generate_synthetic_data(
        dist_type=dist_type, n_samples=2000, seed=42, **params
    )

    # 4. Prepare data for saving
    data = pd.DataFrame(
        {
            "score": y,  # Thickness values
            "group": 1,  # Constant group value
            "age": X.flatten(),  # Flatten ages to 1D array
        }
    )

    # 5. Save to CSV
    csv_filename = f"data_{dist_type}.csv"
    data.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    # 4. Fit GPR and visualize
    # run_gpr(X, y, n_restarts=5, dist_type=dist_type)
