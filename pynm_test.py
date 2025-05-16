import sys
import os


current_dir = "/Users/ayushchaudhary/models_review"
print(current_dir)
functions_path = os.path.join(current_dir, "GAMLSS-python", "functions")
print(functions_path)
sys.path.append(functions_path)

from gamlass_main import Gamlss
from gamlass_main import base

# Create an instance of the Gamlss class
model = Gamlss(model_name="my_model", x_vals=["age", "score"], y_val="score")

# Fit the model
model.fit(
    r_code="""
    model <- gamlss(y ~ pb(age),
        sigma.formula = ~ pb(age),
        nu.formula = ~ 1,
        tau.formula = ~ 1,
        random = ~ 1|site,
        family = SHASH(),
        method = RS(50)
    )
    """,
    data=df,
)

# Get model summary
model_summary = base.summary(robjects.r["my_model"])
print(model_summary)

# # Generate diagnostic plots
# model.plot_diagnostics()

# # Calculate Z-scores
# z_scores = model.z_score(test_data, train_data)

# # Plot Z-scores
# model.plot_z_scores(test_data, train_data, affected=test_data['condition'])

# # Save the model
# model.save_model("path/to/save/model.rds")

# # Load the model
# loaded_model = Gamlss.load_model("path/to/save/model.rds")
