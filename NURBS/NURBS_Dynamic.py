import numpy as np                                        # array handling 
import matplotlib.pyplot as plt                           # plotting 
import matplotlib.animation as animation                  # animation framework 
from geomdl import fitting, NURBS, operations, utilities             # NURBS fitting & ops
from geomdl.visualization import VisMPL                   # curve visualization
import time

#### NOTE: this currently only fits one single time, and then just convects control points according to the prescribed velocity field  #####

# ─────────────────────────────────────────────────────────────────────────────
# 1) Initialize the geometry
xc, yc = 0.5, 0.75  # match Kratos test "ellipse" center
a, b = 0.15, 0.15   # ellipse radii (major and minor semiaxes equivalent, hence circle)
# ─────────────────────────────────────────────────────────────────────────────

# 2) Sample the ellipse contour discretely and interpolate a closed NURBS curve
# This sets the number of discrete points to sample along the ellipse
num_samples = 300 # also controls the number of control points used later on
theta = np.linspace(0, 2*np.pi, num_samples, endpoint=True)  # sample angles, full circle
# generates a list of (x, y) coordinates along an ellipse
pts = [(xc + a*np.cos(t), yc + b*np.sin(t)) for t in theta]   # ellipse points

# Interpolate a closed NURBS curve
# sets the degree of the NURBS curve
degree = 2
# calls the NURBS approximate fitting function from geomdl.fitting
curve = fitting.approximate_curve(pts, degree, ctrlpts_size=300, periodic=True) # global closed NURBS enforced through periodic=True
# sets the evaluation resolution of the NURBS curve for evaluating points later on
curve.delta = 0.01                                           # evaluation resolution


# Visualization setup (static)
curve.vis = VisMPL.VisCurve2D(ctrlpts=True)  # show control points in the visualization 

# 3) Velocity field definition
# AW 28.5: updated to be in accordance with Kratos velo field
# time-dependent, incompressible velocity field, which resembles a vortex-like pattern
def velocity(x, y, t):
    ux = -2 * (np.sin(np.pi * x))**2 * np.sin(np.pi * y) * np.cos(np.pi * y) * np.cos(np.pi * t / 3.0)
    uy =  2 * (np.sin(np.pi * y))**2 * np.sin(np.pi * x) * np.cos(np.pi * x) * np.cos(np.pi * t / 3.0)
    return ux, uy


## Area computation
# ─────────────────────────────────────────────────────────────────────────────
# Define Shoelace (polygon area) function
def polygon_area(coords):
    """
    Computes the area of a closed polygon via the shoelace formula.
    coords: list of [x, y] pairs ordered around the boundary.
    """
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0 
# ─────────────────────────────────────────────────────────────────────────────
# compute initial area
curve.evaluate()                           # ensure evalpts is populated
# Calls the shoelace function on the evaluated points to compute the initial enclosed area of the curve
initial_area = polygon_area(curve.evalpts) # compute A0
# AW 28.5: updated to be in accordance with Kratos sim
# Stores the initial evaluated points as a NumPy array so they can later be compared (e.g., to compute shape drift)
initial_shape = np.array(curve.evalpts)    
# ─────────────────────────────────────────────────────────────────────────────


# 4) Animation parameters
dt = 0.0001
# AW 28.5: enforced total frames to consider 3s sim time
total_frames = int(3.0 / dt) + 1  # ensure t ∈ [0, 3]
refine_every = 20 # knot vector refinement currently deactivated

#  status print every N frames
print_interval = 20  # adjust as desired 
# logs the start time to measure elapsed time for performance tracking
start_time = time.time()
                                      # record start 
# Precompute Greville parameters
p        = curve.degree
kv       = curve.knotvector
# Greville abscissae are specific parameter values used for:
#Evaluating points,
# Placing control points,
# Computing normals, derivatives, etc.
greville = [sum(kv[i+1 : i+1+p]) / p for i in range(len(curve.ctrlpts))] # also not used in current implementation

# 5) Matplotlib figure & artists
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# Forces the plot to have equal aspect ratio, so circles don’t look like ellipses.
ax.set_aspect('equal', 'box')
line, = ax.plot([], [], 'b-', lw=2)
pts_scatter = ax.scatter([], [], c='r', s=20)


# Defines a setup function required by matplotlib.animation.FuncAnimation. It resets all visual elements
def init():
    """Initialize empty frame."""
    # Clears the curve line (no x, y data yet)
    line.set_data([], [])
    # Clears the scatter plot of control points. np.empty((0,2)) creates an empty 2D array with shape (0, 2)
    pts_scatter.set_offsets(np.empty((0,2)))  # clear scatter
    # Returns the initialized visual elements to the animation engine.
    return line, pts_scatter
    
# 4th order runge-kutta convection in time
def rk4_step(x, y, t, dt, velocity):
    k1x, k1y = velocity(x, y, t)
    k2x, k2y = velocity(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, t + 0.5 * dt)
    k3x, k3y = velocity(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, t + 0.5 * dt)
    k4x, k4y = velocity(x + dt * k3x, y + dt * k3y, t + dt)
    
    x_new = x + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
    
    return x_new, y_new
    
# Forward Euler step
def euler_step(x, y, t, dt):
    vx, vy = velocity(x, y, t)
    return x + dt * vx, y + dt * vy
    


# Defines the function to be called once per animation frame
def update(frame):
    # AW 28.5: added this line for adjusted output matching Kratos
    # global initial_shape allows access to the original curve shape (defined earlier) for drift comparison
    global initial_shape
    """Advect control points, refine knots, and update plot."""
    # Computes current simulation time.
    t = frame * dt
    # 5.1 Advect control points
    # Create a list to store the updated control points
    new_ctrlpts = []
    
    # Loop over each control point of the NURBS curve
    # Loop over each control point of the NURBS curve
    for i, (x, y) in enumerate(curve.ctrlpts):
        if use_rk4:
            x_new, y_new = rk4_step(x, y, t, dt)
        else:
            x_new, y_new = euler_step(x, y, t, dt)
        new_ctrlpts.append([x_new, y_new])


        
        # # Alternative: Advect only in normal direction   IT'S NOT A GOOD IDEA
        # Get unit normal at Greville parameter
        # u = greville[i]
        # # Option A: use geomdl.operations.normal
        # # _, (nx, ny) = operations.normal(curve, [u])
        # # Option B: manual (uncomment if you prefer)
        # ders = curve.derivatives(u, order=1)
        # dx, dy = ders[1][0], ders[1][1]
        # tnorm = np.hypot(dx, dy)
        # tx, ty = dx/tnorm, dy/tnorm
        # nx, ny = -ty, tx
        # # Project velocity onto normal
        # dot = vx*nx + vy*ny
        # vnorm = (dot*nx, dot*ny)
        # new_ctrlpts.append([x + dt*vnorm[0], y + dt*vnorm[1]])
    
    # Overwrite the curve’s control points with the advected ones
    curve.ctrlpts = new_ctrlpts  # update control net

    # If enabled, this would insert new knots into the NURBS curve every refine_every steps. It's turned off for now, likely for simplicity or performance
    # 5.2 Knot refinement
    #if (frame+1) % refine_every == 0:
    #    operations.refine_knotvector(curve, [1])             # insert mid-span knots. LETS KEEP IT OFF FOR NOW

    # 5.3 Evaluate and plot
    # Get the evaluated (x, y) points of the updated curve.
    eval_pts = np.array(curve.evalpts)  # list of [x,y] 
    # Compute current area via shoelace
    current_area = polygon_area(eval_pts)
    # Compute relative area error (in %)
    rel_area = (current_area / initial_area - 1.0) * 100
    # Update the curve line plot
    line.set_data(eval_pts[:,0], eval_pts[:,1])
    pts_scatter.set_offsets(curve.ctrlpts)
    # Update the plot title and axis limits
    ax.set_title(f'Time = {t:.2f}, Area error = {rel_area:.2f}%')  # dynamic title
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
   


    # Status print every print_interval frames
    if (frame + 1) % print_interval == 0 or frame == total_frames - 1:
        elapsed = time.time() - start_time    # elapsed time        
        print(
            f"Step {frame+1}/{total_frames}, "
            f"t={t:.2f}s, "
            f"elapsed={elapsed:.2f}s, "
            f"area={current_area:.6f}, "
            f"Δarea={rel_area:+.2f}%"
        )  # f-string formatting

        # AW 28.5: added this for adjusted output matching Kratos
        current_shape = np.array(curve.evalpts)
        shape_error = np.mean(np.linalg.norm(current_shape - initial_shape, axis=1))
        print(f"Current SHAPE ERROR (avg drift): {shape_error:.6e}")
        with open("shape_error.csv", "a") as f:
            f.write(f"{t:.2f},{shape_error:.6e},{rel_area:.6f}\n")
            
    if abs(t - 3.0) < 1e-8:
        np.savetxt("Nurbs_final.csv", eval_pts, delimiter=",", header="X,Y", comments='')

        # 7) Plot initial and final interface in one static PNG
        final_shape = np.array(eval_pts)  # we already have it
        plt.figure(figsize=(6,6))
        plt.plot(initial_shape[:,0], initial_shape[:,1], 'r--', label='Initial Interface', linewidth=2)
        plt.plot(final_shape[:,0], final_shape[:,1], 'b-', label='Final Interface', linewidth=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Initial vs. Final Interface after Convection")
        plt.axis("equal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("comparison_initial_vs_final.png")
        plt.close()  # <--- Instead of plt.show()


    	



    return line, pts_scatter


use_rk4 = False

# 6) Create and save animation
anim = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=total_frames, interval=50, blit=False      # blit=False to avoid backend issues
)

# Use FFmpegWriter for MP4 output
writer = animation.FFMpegWriter(fps=1)               # requires FFmpeg installed 
anim.save('vortex_interface.mp4', writer=writer)  #, save_count=total_frames save to MP4

plt.close(fig)
