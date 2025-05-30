## Description
The GAMLSS package provides a framework for fitting and analyzing generalized additive models for location, scale, and shape. It allows for modeling the response variable's mean, variance, skewness, and kurtosis as functions of explanatory variables using various distributions and link functions.

This Python package integrates with the R programming language using the `rpy2` library to leverage the functionality of the GAMLSS package in R.

## Installation
### 1. Set up the Python environment
1. Make sure you have Anaconda or Miniconda installed on your system.
2. Create a new conda environment:
   ```bash
   conda create -n gamlss python=3.10.15
   conda activate gamlss
   ```

3. Install required Python packages using the provided requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### 2. R Requirements (It is already installed for MHMLab - so skip please)
Ensure R (>= 4.1.2) is installed with the `gamlss` package. Install gamlss in R using:
```R
install.packages("gamlss")
```

Now you're ready to use the GAMLSS package in Python!

## Usage
Here's a basic example of how to use the GAMLSS package:

```python
from gamlass_main import Gamlss

# Create an instance of the Gamlss class
model = Gamlss(model_name='my_model', x_vals=['age', 'sex', 'site'], y_val='outcome')

# Fit the model
model.fit(
    r_code="""
    model <- gamlss(y ~ pb(age, by = as.factor(sex)) + site,
        sigma.formula = ~ pb(age) + site,
        nu.formula = ~ 1,
        tau.formula = ~ 1,
        random = ~ 1|site,
        family = SHASH(),
        method = RS(50)
    )
    """,
    data=df_train
)

# Get model summary
model_summary = base.summary(robjects.r[model_name])
print(model_summary)

# Generate diagnostic plots
model.plot_diagnostics()

# Calculate Z-scores
z_scores = model.z_score(test_data, train_data)

# Plot Z-scores
model.plot_z_scores(test_data, train_data, affected=test_data['condition'])

# Save the model
model.save_model("path/to/save/model.rds")

# Load the model
loaded_model = Gamlss.load_model("path/to/save/model.rds")
```

For more detailed usage and examples, please refer to the [Notion GAMLSS Documentation](https://www.notion.so/Documentation-of-GAMLSS-python-package-e0f3230be6f745c1940da228af859977/) and examples in the file quick_module_use.ipynb

## Features
- Fit GAMLSS models using various distributions and link functions
- Predict from fitted GAMLSS models
- Calculate and plot Z-scores for test data
- Generate and visualize centile plots
- Perform model comparisons based on AIC and BIC
- Conduct permutation testing for robust model evaluation
- Create diagnostic plots for model assessment
- Parallel processing support for efficient computation
- Caching mechanism for storing and reusing model results
- Save and load fitted models for later use
- Site transfer functionality for testing new sites without retraining

## Advanced Features (Updated Version)
- Parallel Processing: Utilize multiple CPU cores for faster computation of multiple models.
- Caching: Store and reuse model results to save time on repeated analyses.
- Brain Visualization: Map statistical results onto brain surface models for intuitive interpretation.
- Model Saving and Loading: Save fitted models and load them later for predictions or further analysis.
- Flexible Multiprocessing: Option to disable multiprocessing if issues arise.
- Enhanced Formula Comparison: Compare different formulas with multiprocessing and model saving capabilities.

## Acknowledgments

- The GAMLSS package in R: [https://www.gamlss.com/](https://www.gamlss.com/)
- The `rpy2` library: [https://rpy2.github.io/](https://rpy2.github.io/)

