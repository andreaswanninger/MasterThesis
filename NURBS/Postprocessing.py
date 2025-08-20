import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
# Utility: Shoelace area formula
def polygon_area(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) # minimal implementation of shoelace formula using np.roll and vectorized form instead of looping over all points

# Utility: Sort points by angle from centroid
def sort_points(coords):
    centroid = np.mean(coords, axis=0) # geometric center of the point cloud
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0]) # assigns each point an angle from 0 to 2*pi
    return coords[np.argsort(angles)] # gives back point set sorting clockwise/counter-clockwise

# Utility: Resample closed curves with uniform pointwise distances and number of points
def resample_closed_curve(points, num=500):
    points = np.asarray(points)
    diffs = np.diff(np.vstack([points, points[0]]), axis=0) # returns pointwise differences between points
    seg_lengths = np.linalg.norm(diffs, axis=1) # using Euclidean distance, computes the length of each segment
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)
    cumulative /= cumulative[-1]  # normalize to [0,1]; cumulative tells us how far we are along the contour (ranging from 0 to 1)

    interp_x = interp1d(cumulative, np.append(points[:, 0], points[0, 0]), kind='linear')
    interp_y = interp1d(cumulative, np.append(points[:, 1], points[0, 1]), kind='linear')

    uniform_s = np.linspace(0, 1, num, endpoint=False) # creates uniform number of locations parametrized along the curve
    return np.column_stack((interp_x(uniform_s), interp_y(uniform_s)))

# Utility: Symmetric shape error
# computesm geometric difference between two shapes (represented as 2D point clouds)
# --> uses a symmetric nearest-neighbor distance metric
# idea: measure how close one curve is to the other and the other way around, and divide by two
def shape_error(reference_analytic, target_points):
    reference = np.asarray(reference_analytic)
    target = np.asarray(target_points)

    tree_ref = cKDTree(reference) # KD-tree (k-dimensional tree) allows efficient nearest-neighbor queries in space
    tree_tgt = cKDTree(target)

    dist_ref_to_tgt, _ = tree_tgt.query(reference)
    dist_tgt_to_ref, _ = tree_ref.query(target)

    return 0.5 * (np.mean(dist_ref_to_tgt) + np.mean(dist_tgt_to_ref)) # effectively, simple implementation of Hausdorff matrix

# Utility: Radial deviation from perfect circle
def radial_deviation_error(points, x0, y0, R_expected):
    distances = np.linalg.norm(points - np.array([x0, y0]), axis=1) # computes the distance of every point to the geometric center (using Euclidean distance)
    return np.mean(np.abs(distances - R_expected))

# Utility: Visualize radius error
def plot_radius_deviation(points, x0, y0, R_expected, title, output_name):
    distances = np.linalg.norm(points - np.array([x0, y0]), axis=1)
    deviation = distances - R_expected
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(points[:, 0], points[:, 1], c=deviation, cmap='coolwarm', s=5)
    plt.colorbar(sc, label="Radial deviation [m]")
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

# ─────────────────────────────────────────────
# 1. Load extracted isosurface
df_extracted = pd.read_csv("cls_64.csv")
x_extracted = df_extracted["Points:0"].dropna().values
y_extracted = df_extracted["Points:1"].dropna().values
extracted = np.column_stack((x_extracted, y_extracted))
extracted_sorted = sort_points(extracted) # sort points according

# 2. Load final NURBS interface
df_final = pd.read_csv("Nurbs_final.csv")
x_final = df_final["X"].values
y_final = df_final["Y"].values
final = np.column_stack((x_final, y_final))
final_sorted = sort_points(final)

# ─────────────────────────────────────────────
# 3. Analytical circle reference
Xc, Yc = 0.5, 0.75
R_ref = 0.15
theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
x_circle = Xc + R_ref * np.cos(theta)
y_circle = Yc + R_ref * np.sin(theta)
reference_circle = np.column_stack((x_circle, y_circle))

# ─────────────────────────────────────────────
# 4. Area comparison (original points only)
area_ref = polygon_area(reference_circle)
area_extracted = polygon_area(extracted_sorted)
area_final = polygon_area(final_sorted)
err_area_extracted = 100 * (area_extracted - area_ref) / area_ref # relative error wrt reference
err_area_final = 100 * (area_final - area_ref) / area_ref

# ─────────────────────────────────────────────
# 5. Shape and radial errors (resampled points)
resampled_ref = resample_closed_curve(reference_circle, 500)
resampled_extracted = resample_closed_curve(extracted_sorted, 500)
resampled_final = resample_closed_curve(final_sorted, 500)

shape_err_extracted = shape_error(resampled_ref, resampled_extracted)
shape_err_final = shape_error(resampled_ref, resampled_final)

radial_err_extracted = radial_deviation_error(resampled_extracted, Xc, Yc, R_ref)
radial_err_final = radial_deviation_error(resampled_final, Xc, Yc, R_ref)

# ─────────────────────────────────────────────
# 6. Plot original (unsampled) shapes with shape/area errors in label
plt.figure(figsize=(6, 6))
plt.plot(extracted_sorted[:, 0], extracted_sorted[:, 1], 'b.', markersize=2,
         label=f"Kratos final interface\nArea error = {err_area_extracted:+.2f}%\nSymmetric shape error = {shape_err_extracted:.2e}\nMean Radial deviation = {radial_err_extracted:.2e}")
plt.plot(reference_circle[:, 0], reference_circle[:, 1], 'r--', linewidth=1.5,
         label="Initial Interface (spherical)")
plt.plot(final_sorted[:, 0], final_sorted[:, 1], 'g-', linewidth=1.2,
         label=f"NURBS final interface\nArea error = {err_area_final:+.2f}%\nSymmetric shape error = {shape_err_final:.2e}\nMean Radial deviation = {radial_err_final:.2e}")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend(loc="upper right", fontsize=8)
plt.title("Comparison of Initial (Analytical), Kratos based, and Nurbs based Interface")
plt.tight_layout()
plt.savefig("interface_comparison_with_shape_errors.png", dpi=300)
plt.show()

# ─────────────────────────────────────────────
# 7. Plot radial deviation maps (resampled points)
plot_radius_deviation(resampled_extracted, Xc, Yc, R_ref,
                      "Radial Deviation - Kratos Final Interface",
                      "Radial Deviation_Kratos.png")

plot_radius_deviation(resampled_final, Xc, Yc, R_ref,
                      "Radial Deviation - NURBS Final Interface",
                      "radial_deviation_Nurbs.png")

# ─────────────────────────────────────────────
# 8. Extra Visualization: Shoelace Area (Kratos)
plt.figure(figsize=(6, 6))
plt.fill(extracted_sorted[:, 0], extracted_sorted[:, 1], color='skyblue', alpha=0.4, label="Kratos Extracted")
plt.plot(extracted_sorted[:, 0], extracted_sorted[:, 1], 'b.-', markersize=2)
plt.title(f"Shoelace Area (Kratos) = {area_extracted:.5e} m²")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig("shoelace_area_Kratos.png", dpi=300)
plt.close()

# 9. Extra Visualization: Shoelace Area (NURBS)
plt.figure(figsize=(6, 6))
plt.fill(final_sorted[:, 0], final_sorted[:, 1], color='lightgreen', alpha=0.4, label="NURBS Final")
plt.plot(final_sorted[:, 0], final_sorted[:, 1], 'g.-', markersize=2)
plt.title(f"Shoelace Area (NURBS) = {area_final:.5e} m²")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig("shoelace_area_Nurbs.png", dpi=300)
plt.close()

# 10. Extra Visualization: Resampled vs Original (Kratos)
plt.figure(figsize=(6, 6))
plt.plot(extracted_sorted[:, 0], extracted_sorted[:, 1], 'b-', label="Original")
plt.plot(resampled_extracted[:, 0], resampled_extracted[:, 1], 'ro', markersize=2, label="Resampled")
plt.title("Original vs Resampled - Kratos Extracted")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Kratos_resampled_vs_original.png", dpi=300)
plt.close()

# 11. Extra Visualization: Resampled vs Original (NURBS)
plt.figure(figsize=(6, 6))
plt.plot(final_sorted[:, 0], final_sorted[:, 1], 'g-', label="Original")
plt.plot(resampled_final[:, 0], resampled_final[:, 1], 'ro', markersize=2, label="Resampled")
plt.title("Original vs Resampled - NURBS Final")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("NURBS_resampled_vs_original.png", dpi=300)
plt.close()

