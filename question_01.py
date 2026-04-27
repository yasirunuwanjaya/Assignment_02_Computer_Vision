import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
try:
    D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading lines.csv: {e}")
    exit()

# ---------------------------------------------------------
# Part (a): Total Least Squares (TLS) for Line 1
# ---------------------------------------------------------
# Extracting first line coordinates (x1, y1)
x1 = D[:, 0]
y1 = D[:, 3]
pts1 = np.vstack([x1, y1]).T

def total_least_squares(points):
    centroid = np.mean(points, axis=0)
    centered_pts = points - centroid
    # SVD decomposition
    _, _, Vt = np.linalg.svd(centered_pts)
    # The normal to the line is the vector corresponding to the smallest singular value
    a, b = Vt[-1]
    # Equation: ax + by + d = 0 -> d = -(ax_mean + by_mean)
    d = -(a * centroid[0] + b * centroid[1])
    
    # Convert to y = mx + c for reporting
    m = -a / b
    c = -d / b
    return m, c

m_tls, c_tls = total_least_squares(pts1)
print(f"\n--- Part (a) ---")
print(f"TLS Parameters for Line 1: Slope (m) = {m_tls:.4f}, Intercept (c) = {c_tls:.4f}")

# ---------------------------------------------------------
# Part (b): Sequential RANSAC for 3 Lines
# ---------------------------------------------------------
# Flatten all points as indicated in your code snippet
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten().reshape(-1, 1)
Y_all = Y_cols.flatten()

print(f"\n--- Part (b) ---")
remaining_X = X_all
remaining_Y = Y_all
colors = ['r', 'g', 'b'] # Colors for plotting

plt.figure(figsize=(10, 6))
plt.scatter(X_all, Y_all, color='gray', alpha=0.3, label='Original Points')

for i in range(3):
    # Initialize RANSAC
    # residual_threshold determines how close a point must be to be an 'inlier'
    ransac = RANSACRegressor(residual_threshold=0.5, random_state=42)
    ransac.fit(remaining_X, remaining_Y)
    
    # Get parameters
    m_ran = ransac.estimator_.coef_[0]
    c_ran = ransac.estimator_.intercept_
    print(f"RANSAC Line {i+1}: y = {m_ran:.4f}x + {c_ran:.4f}")
    
    # Identify inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask
    
    # Plotting the found line
    line_x = np.array([X_all.min(), X_all.max()])
    line_y = m_ran * line_x + c_ran
    plt.plot(line_x, line_y, color=colors[i], lw=2, label=f'Line {i+1} Fit')
    plt.scatter(remaining_X[inlier_mask], remaining_Y[inlier_mask], color=colors[i], s=10)

    # Update the dataset by removing the points we just "explained"
    remaining_X = remaining_X[outlier_mask]
    remaining_Y = remaining_Y[outlier_mask]

plt.title("Sequential RANSAC Multi-Line Fitting")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.legend()
plt.savefig("ransac_result.png") # This saves the image to your folder
plt.show()

print("\nProcessing complete. Check 'ransac_result.png' for the visualization.")