import math

# Prompt user for input
theta_deg = float(input("Enter the contact angle θ (in degrees): "))
d = float(input("Enter the domain width d (in meters): "))

# Convert angle to radians
angle_rad = math.radians(theta_deg - 90)

# Compute radius
R = d / (2 * math.sin(angle_rad))

# Display result
print(f"\nComputed radius R = {R:.6f} m")
print(f"\nComputed curvature kappa = {1/R:.6f} m-1")

