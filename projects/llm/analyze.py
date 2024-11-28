import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the CSV file
file_path = "./results_summary.csv"
data = pd.read_csv(file_path)

# Fill NaN values in the relevant feature columns with False
data_filled = data.copy()

# Remove rows with missing target variable (word_perplexity)
data_filled.dropna(subset=["word_perplexity"], inplace=True)

# Extract features and target variable
features = [
    "quant.enable_rotation",
    "quant.enable_reorder",
    "quant.smooth.enable_xw",
    "quant.smooth.enable_yx",
    "quant.wgts.enable_calib_range",
]
X = data_filled[features]
y = data_filled["word_perplexity"]

# Convert boolean columns to numeric for regression
X = X.astype(int)

# Add a constant for the regression model
X = sm.add_constant(X)

# Fit a linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
model_summary = model.summary()
print(model_summary)

import matplotlib.pyplot as plt
import seaborn as sns

# Feature name mapping for better visualization
feature_names = {
    'quant.enable_rotation': 'rotation',
    'quant.enable_reorder': 'reorder',
    'quant.smooth.enable_xw': 'awq_smooth',
    'quant.smooth.enable_yx': 'smooth attention',
    'quant.wgts.enable_calib_range': 'awq_calib'
}

# Update coefficients and errors for plotting
coefficients = model.params[1:]  # Exclude constant
errors = model.bse[1:]           # Standard errors
p_values = model.pvalues[1:]     # p-values
labels = [feature_names[feature] for feature in coefficients.index]


# Plot with updated labels and p-values
plt.figure(figsize=(10, 6))
# sns.barplot(x=labels, y=coefficients.values, errorbar=None, palette="coolwarm")
# Normalize Y values to map to colors
norm = plt.Normalize(coefficients.min(), coefficients.max())
colors = cm.coolwarm(norm(coefficients))

# Plot the barplot with custom colors based on Y values
plt.figure(figsize=(10, 6))
bars = plt.bar(x=labels, height=coefficients.values, yerr=errors, color=colors, capsize=5, edgecolor='black')

plt.errorbar(x=range(len(coefficients)), y=coefficients, yerr=errors, fmt='o', color='black', capsize=5)

# Add grid and customize ticks
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(range(len(labels)), labels, rotation=0, ha='center', fontsize=12)
plt.ylabel("Affect on the Perplexity", fontsize=14)
plt.title("Impact of Features on Word Perplexity", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Customize the graph
#plt.title("Impact of Features on Word Perplexity with Significance", fontsize=16)
plt.xlabel("Feature", fontsize=14)
plt.tight_layout()
plt.savefig("figures.pdf", format="pdf")

