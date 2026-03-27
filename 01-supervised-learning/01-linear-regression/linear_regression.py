"""
Linear Regression — From Scratch and with Scikit-learn

This script demonstrates linear regression in two ways:
1. Implementing the algorithm from scratch using the Normal Equation
2. Using scikit-learn's LinearRegression for the same task

Both approaches find the line of best fit: y = b₀ + b₁x
where b₀ is the intercept and b₁ is the slope.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Generate Synthetic Data ---
# We create data with a known linear relationship: y = 3 + 4x + noise
# Knowing the "true" values lets us check how well the model recovers them.
np.random.seed(42)
X = 2 * np.random.rand(100, 1)                 # 100 data points, values between 0 and 2
y = 3 + 4 * X + np.random.randn(100, 1)        # true intercept=3, true slope=4, plus noise

# --- 2. Split into Training and Test Sets ---
# This is critical: we train the model on one portion of the data
# and evaluate it on data it has never seen. This tests generalisation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- 3. Linear Regression from Scratch (Normal Equation) ---
# The Normal Equation finds the optimal parameters in one step:
#   θ = (Xᵀ X)⁻¹ Xᵀ y
# We need to add a column of 1s to X for the intercept term.
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]  # add bias column (x₀ = 1)
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

intercept_scratch = theta[0][0]
slope_scratch = theta[1][0]

print("\n--- From Scratch (Normal Equation) ---")
print(f"Intercept (b₀): {intercept_scratch:.4f}")
print(f"Slope (b₁):     {slope_scratch:.4f}")

# --- 4. Linear Regression with Scikit-learn ---
# Scikit-learn does the same thing behind the scenes,
# but with numerical optimisations for stability and speed.
model = LinearRegression()
model.fit(X_train, y_train)

intercept_sklearn = model.intercept_[0]
slope_sklearn = model.coef_[0][0]

print("\n--- Scikit-learn ---")
print(f"Intercept (b₀): {intercept_sklearn:.4f}")
print(f"Slope (b₁):     {slope_sklearn:.4f}")
print(f"\n(True values: intercept=3, slope=4)")

# --- 5. Evaluate on the Test Set ---
# Predictions on data the model has never seen.
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Test Set Evaluation ---")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error:   {rmse:.4f}")
print(f"R² Score:                  {r2:.4f}")

# --- 6. Predict on New, Unseen Data ---
# The practical use case: given a new X value, what does the model predict?
X_new = np.array([[0.5], [1.0], [1.5], [2.0]])
y_new_pred = model.predict(X_new)

print("\n--- Predictions on New Data ---")
for x_val, y_val in zip(X_new.flatten(), y_new_pred.flatten()):
    print(f"  X = {x_val:.1f}  →  ŷ = {y_val:.2f}  (true y ≈ {3 + 4 * x_val:.2f})")

# --- 7. Visualise the Results ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: regression line against data
ax1 = axes[0]
ax1.scatter(X_train, y_train, color='steelblue', alpha=0.6, label='Training data')
ax1.scatter(X_test, y_test, color='coral', alpha=0.6, marker='s', label='Test data')

# Plot regression line using sorted values for a clean line
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, color='black', linewidth=2, label=f'ŷ = {intercept_sklearn:.2f} + {slope_sklearn:.2f}x')

ax1.set_title('Linear Regression — Fit')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right panel: residuals (prediction errors)
residuals = y_test - y_pred
ax2 = axes[1]
ax2.scatter(y_pred, residuals, color='steelblue', alpha=0.6)
ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax2.set_title('Residual Plot — Test Set')
ax2.set_xlabel('Predicted ŷ')
ax2.set_ylabel('Residual (y − ŷ)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to linear_regression_results.png")
