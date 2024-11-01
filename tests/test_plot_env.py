# cylinders.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('TkAgg')

class Cylinder:
    def __init__(self, x, y, z, direction, height, r):
        self.center = np.array([x, y, z])
        self.height = height
        self.radius = r
        self.direction = direction / np.linalg.norm(direction)
        self.bottom_center = self.center - 0.5 * self.height * self.direction
        self.top_center = self.center + 0.5 * self.height * self.direction

    def is_point_inside(self, x, y, z):
        p = np.array([x, y, z])
        p_minus_c = p - self.center
        t = np.dot(p_minus_c, self.direction)
        closest_point_on_axis = self.center + t * self.direction
        distance = np.linalg.norm(p - closest_point_on_axis)
        return abs(t) <= 0.5 * self.height and distance <= self.radius

    def draw(self, ax, color='b'):
        theta = np.linspace(0, 2 * np.pi, 10)
        z = np.linspace(0, self.height, 10)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)

        rot_axis = np.cross([0, 0, 1], self.direction)
        rot_angle = np.arccos(np.dot([0, 0, 1], self.direction))

        rotation = R.from_rotvec(rot_axis * rot_angle)
        xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        xyz_rotated = rotation.apply(xyz.T).T

        X, Y, Z = xyz_rotated
        X = X.reshape(x.shape) + self.bottom_center[0]
        Y = Y.reshape(y.shape) + self.bottom_center[1]
        Z = Z.reshape(z.shape) + self.bottom_center[2]

        ax.plot_surface(X, Y, Z, color=color, alpha=0.6)
        self.draw_caps(ax, rotation, color)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def draw_caps(self, ax, rotation, color):
        theta = np.linspace(0, 2 * np.pi, 30)
        circle = np.vstack([self.radius * np.cos(theta), self.radius * np.sin(theta), np.zeros_like(theta)])
        circle_rotated = rotation.apply(circle.T).T

        Xb, Yb, Zb = circle_rotated
        Xb = Xb + self.bottom_center[0]
        Yb = Yb + self.bottom_center[1]
        Zb = Zb + self.bottom_center[2]
        verts = [list(zip(Xb, Yb, Zb))]
        ax.add_collection3d(art3d.Poly3DCollection(verts, color=color, alpha=0.6))

        Xt, Yt, Zt = circle_rotated
        Xt = Xt + self.top_center[0]
        Yt = Yt + self.top_center[1]
        Zt = Zt + self.top_center[2]
        verts = [list(zip(Xt, Yt, Zt))]
        ax.add_collection3d(art3d.Poly3DCollection(verts, color=color, alpha=0.6))

cylinder_params = {
    "platform1": {"x": 1, "y": 0, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "skyblue"},
    "platform2": {"x": -1, "y": 0, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "skyblue"},
    "connector1": {"x": 0, "y": 0, "z": 0.3, "direction": np.array([1, 0, 0]), "height": 2, "r": 0.1, "color": "skyblue"},

    "platform3": {"x": 0, "y": 1, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "skyblue"},
    "platform4": {"x": 0, "y": -1, "z": 0.3, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "skyblue"},
    "connector2": {"x": 0, "y": 0, "z": 0.3, "direction": np.array([0, 1, 0]), "height": 2, "r": 0.1, "color": "skyblue"},

    "platform5": {"x": 1, "y": 0, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "lightgreen"},
    "platform6": {"x": -1, "y": 0, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "lightgreen"},
    "connector3": {"x": 0, "y": 0, "z": 1.4, "direction": np.array([1, 0, 0]), "height": 2, "r": 0.1, "color": "lightgreen"},

    "platform7": {"x": 0, "y": 1, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "lightgreen"},
    "platform8": {"x": 0, "y": -1, "z": 1.4, "direction": np.array([0, 0, 1]), "height": 0.2, "r": 0.2, "color": "lightgreen"},
    "connector4": {"x": 0, "y": 0, "z": 1.4, "direction": np.array([0, 1, 0]), "height": 2, "r": 0.1, "color": "lightgreen"},

    "center_conncection": {"x": 0, "y": 0, "z": 0.8, "direction": np.array([0, 0, 1]), "height": 1.6, "r": 0.1, "color": "lightcoral"},
    "center_base": {"x": 0, "y": 0, "z": 0.05, "direction": np.array([0, 0, 1]), "height": 0.1, "r": 0.4, "color": "lightcoral"},
}

# Grid parameters for dest_area
grid_params = {
    "center": np.array([0, 0]),  # Center of the grid
    "z": 4.5,                      # Height of the grid
    "spacing": 1.0,              # Spacing between points
    "size": 4,                   # Size of the grid (4x4)
    "color": "lightcoral"             # Color of the grid points
}

def draw_grid(ax, center, z, spacing=1.0, size=4, color='lightgrey'):
    half_size = (size - 1) / 2 * spacing
    x_vals = np.arange(-half_size, half_size + spacing, spacing) + center[0]
    y_vals = np.arange(-half_size, half_size + spacing, spacing) + center[1]
    for x in x_vals:
        for y in y_vals:
            ax.scatter(x, y, z, color=color)

def create_cylinders(ax):
    for key, params in cylinder_params.items():
        cylinder = Cylinder(params["x"], params["y"], params["z"], params["direction"], params["height"], params["r"])
        cylinder.draw(ax, color=params["color"])

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    create_cylinders(ax)
    ax.set_box_aspect([2, 2, 1])
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 4])

    # Draw the grid on dest_area using grid_params
    draw_grid(ax, **grid_params)

    plt.show()
