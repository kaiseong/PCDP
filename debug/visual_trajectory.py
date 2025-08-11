import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the data
try:
    df = pd.read_csv('RISE_action_data.csv')
except FileNotFoundError:
    print("eef_data.csv not found. Make sure the file is in the same directory as the script.")
    exit()

# --- Figure 1: Trajectory with Color Gradient ---
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
fig1.canvas.manager.set_window_title('Trajectory Plot')

n_points = len(df)
x_traj, y_traj, z_traj = df['x'], df['y'], df['z']

# Create a colormap for the trajectory line and points
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Plot the trajectory line segment by segment
for i in range(n_points - 1):
    ax1.plot(x_traj[i:i+2], y_traj[i:i+2], z_traj[i:i+2], color=colors[i])

# Add a scatter plot for the points
sc = ax1.scatter(x_traj, y_traj, z_traj, c=np.arange(n_points), cmap='viridis', s=10)

# Add a color bar
cbar = fig1.colorbar(sc, ax=ax1, shrink=0.6)
cbar.set_label('Time Progression (Old to New)')

# Draw world axes for reference
ax1.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1, arrow_length_ratio=0.1, label='World X-axis')
ax1.quiver(0, 0, 0, 0, 1, 0, color='b', length=0.1, arrow_length_ratio=0.1, label='World Y-axis')
ax1.quiver(0, 0, 0, 0, 0, 1, color='g', length=0.1, arrow_length_ratio=0.1, label='World Z-axis')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('End Effector Trajectory')
ax1.legend()
ax1.set_aspect('equal', adjustable='box')


# --- Figure 2: Coordinate Frames with Transparency ---
fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection='3d')
fig2.canvas.manager.set_window_title('Orientation Plot')

axis_colors = {'x': 'r', 'y': 'b', 'z': 'g'}
axis_length = 0.02
alphas = np.linspace(0.1, 1.0, n_points)

# Plot a coordinate frame for each point
for i in range(n_points):
    x, y, z = df.loc[i, ['x', 'y', 'z']]
    roll, pitch, yaw = df.loc[i, ['roll', 'pitch', 'yaw']]
    
    rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    rot_matrix = rot.as_matrix()
    current_alpha = alphas[i]

    for j, axis_name in enumerate(['x', 'y', 'z']):
        direction = rot_matrix[:, j]
        color = axis_colors[axis_name]
        ax2.quiver(x, y, z, 
                   direction[0], direction[1], direction[2],
                   length=axis_length, 
                   color=color, 
                   alpha=current_alpha,
                   arrow_length_ratio=0.3)

# Draw world axes for reference
ax2.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1, arrow_length_ratio=0.1, label='World X-axis')
ax2.quiver(0, 0, 0, 0, 1, 0, color='b', length=0.1, arrow_length_ratio=0.1, label='World Y-axis')
ax2.quiver(0, 0, 0, 0, 0, 1, color='g', length=0.1, arrow_length_ratio=0.1, label='World Z-axis')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('End Effector Orientation as Coordinate Frames')
ax2.legend()
ax2.autoscale_view()
ax2.set_aspect('equal', adjustable='box')

# Show both plots in separate windows
plt.show()