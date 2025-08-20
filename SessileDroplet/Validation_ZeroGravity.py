import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the intersection points from the provided .txt file
data = pd.read_csv("intersection_points.txt", delim_whitespace=True, comment='#')
x_points = data['X'].values
y_points = data['Y'].values

def compute_radius_from_area(A0, theta_rad):
    # Computes the radius of curvature Rc so the droplet has constant area A0 for contact angle theta
    term = theta_rad - 0.5 * np.sin(2 * theta_rad)
    return np.sqrt(A0 / term)

def plot_droplet_with_annotations_and_points(A0, theta_deg_list, x_offset):
    plt.figure(figsize=(9, 4))

    for i, theta_deg in enumerate(theta_deg_list):
        theta_rad = np.radians(theta_deg)
        Rc = compute_radius_from_area(A0, theta_rad)
        h = Rc * (1 - np.cos(theta_rad))
        y_center = -Rc * np.cos(theta_rad)
        L = 2 * Rc * np.sin(theta_rad)

        # Arc coordinates
        phi = np.linspace(-theta_rad, theta_rad, 500)
        x = Rc * np.sin(phi) + x_offset
        y = y_center + Rc * np.cos(phi)

        # Plot droplet
        plt.plot(x, y, label=f'θ={theta_deg}°, Rc={Rc*1e3:.2f} mm, A={A0*1e6:.2f} mm²')

        # Annotate base diameter (L)
        plt.annotate("", xy=(x_offset + L/2, 0), xytext=(x_offset - L/2, 0),
                     arrowprops=dict(arrowstyle='<->', color='black'))
        plt.text(x_offset, -0.00008 - 0.00005*i, f'L = {L*1e3:.2f} mm', ha='center', fontsize=8)

        # Annotate height (h)
        plt.annotate("", xy=(x_offset, 0), xytext=(x_offset, h),
                     arrowprops=dict(arrowstyle='<->', color='black'))
        plt.text(x_offset + 0.0004*i, h/2, f'h = {h*1e3:.2f} mm', fontsize=8)

        print(f"θ = {theta_deg:>3}°, Rc = {Rc:.6e} m, L = {L:.6e} m, h = {h:.6e} m, Area = {A0:.6e} m²")

    # Overlay intersection points
    plt.scatter(x_points, y_points, color='red', s=5, label='Intersection Points')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Sessile Droplet with Fixed Area and Intersection Overlay')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Define constant area A0 using 90° droplet and known radius
theta_90_rad = np.radians(90)
Rc_90 = 0.006  # radius in meters
A0 = Rc_90**2 * (theta_90_rad - 0.5 * np.sin(2 * theta_90_rad))

# Call the function to plot centered at x = 0.015 m
plot_droplet_with_annotations_and_points(A0, [90, 102.5, 105], x_offset=0.015)
