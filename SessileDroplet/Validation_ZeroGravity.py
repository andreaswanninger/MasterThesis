import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_intersection_points(filename):
    data = pd.read_csv(filename, delim_whitespace=True, comment='#')
    return data['X'].values, data['Y'].values

def compute_radius_from_area(A0, theta_rad):
    term = theta_rad - 0.5 * np.sin(2 * theta_rad)
    return np.sqrt(A0 / term)

def plot_droplet_with_annotations_and_points(A0, theta_deg_list, x_offset, output_filename):
    plt.figure(figsize=(9, 4))
    legend_entries = []

    # Map of angle to corresponding intersection file
    intersection_files = {
        75: "intersection_points_75.txt",
        90: "intersection_points_90.txt",
        105: "intersection_points_105.txt"
    }

    colors = {75: 'blue', 90: 'green', 105: 'orange'}  # distinguish intersection points
    point_handles = []

    for theta_deg in theta_deg_list:
        theta_rad = np.radians(theta_deg)
        Rc = compute_radius_from_area(A0, theta_rad)
        h = Rc * (1 - np.cos(theta_rad))
        y_center = -Rc * np.cos(theta_rad)
        L = 2 * Rc * np.sin(theta_rad)
        
        print("Rc: ", Rc)

        # Generate arc
        phi = np.linspace(-theta_rad, theta_rad, 500)
        x = Rc * np.sin(phi) + x_offset
        y = y_center + Rc * np.cos(phi)

        # Droplet shape label (theta, L, h only)
        label = (f'θ={theta_deg}° | '
                 f'L={L*1e3:.2f} mm | h={h*1e3:.2f} mm | R_c={Rc*1e3:.2f} mm')
        legend_entries.append(label)
        plt.plot(x, y)

        # Load and plot intersection points
        if theta_deg in intersection_files:
            x_pts, y_pts = load_intersection_points(intersection_files[theta_deg])
            handle = plt.scatter(x_pts, y_pts, color=colors[theta_deg], s=5, label=f'Intersection Points θ={theta_deg}°')
            point_handles.append(handle)

        print(f"θ = {theta_deg:>5.1f}°, Rc = {Rc:.6e} m, L = {L:.6e} m, h = {h:.6e} m, Area = {A0:.6e} m²")

    # Add all droplet labels manually to the legend
    for entry in legend_entries:
        plt.plot([], [], ' ', label=entry)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Sessile Droplets and Intersection Overlays')
    plt.legend(fontsize=8)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Area from 90° droplet
theta_90_rad = np.radians(90)
Rc_90 = 0.006
A0 = Rc_90**2 * (theta_90_rad - 0.5 * np.sin(2 * theta_90_rad))

# Plot and save
plot_droplet_with_annotations_and_points(
    A0,
    theta_deg_list=[75,90,105],
    x_offset=0.015,
    output_filename='sessileDrop_zeroGravity_ShapeComparison.png'
)
