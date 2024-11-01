import sys
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d

class Sprite:
    def __init__(self, ax, t_lim=30):
        self.t_lim = t_lim
        self.ax = ax

    def update(self, t, done=False):
        raise NotImplemented

    def get_sphere(self, center, radius, color="r", alpha=0.1):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        return self.ax["ax_3d"].plot_wireframe(x, y, z, color=color, alpha=alpha)

    def get_cube(self, vertex, l=1, w=1, h=1, alpha=0.1, color="r"):
        x1, y1, z1 = vertex[0], vertex[1], vertex[2]
        x2, y2, z2 = x1 + l, y1 + w, z1 + h
        ax = self.ax["ax_3d"]
        xs, ys = np.meshgrid([x1, x2], [y1, y2])
        zs = np.ones_like(xs)
        body = []
        body.append(ax.plot_wireframe(xs, ys, zs * z1, alpha=alpha, color=color))
        body.append(ax.plot_wireframe(xs, ys, zs * z2, alpha=alpha, color=color))
        ys, zs = np.meshgrid([y1, y2], [z1, z2])
        xs = np.ones_like(ys)
        body.append(ax.plot_wireframe(xs * x1, ys, zs, alpha=alpha, color=color))
        body.append(ax.plot_wireframe(xs * x2, ys, zs, alpha=alpha, color=color))
        return body

class SphereSprite(Sprite):
    def __init__(self, ax, color="r", alpha=0.1, t_lim=30):
        self.color = color
        self.alpha = alpha
        self.body = None
        super().__init__(ax, t_lim)

    def update(self, position, radius):
        if self.body:
            if isinstance(self.body, list):
                for body in self.body:
                    body.remove()
            else:
                self.body.remove()

        self.body = self.get_sphere(position, radius, color=self.color, alpha=self.alpha)

class ObstacleSprite(SphereSprite):
    def __init__(self, ax, obstacle, t_lim=30):
        super().__init__(ax, t_lim=t_lim)
        self.obstacle = obstacle

    def update(self, t, done=False):
        position = self.obstacle.state[:3]
        radius = self.obstacle.r
        super().update(position, radius)

class UavSprite(Sprite):
    def __init__(self, ax, uav=None, color=None, t_lim=10):
        self.ax = ax
        self.t_lim = t_lim
        self.uav = uav
        self.pad_sprite = None
        self.color = color

        (self.l1,) = self.ax["ax_3d"].plot([], [], [], color=color, linewidth=1, antialiased=False)
        (self.l2,) = self.ax["ax_3d"].plot([], [], [], color=color, linewidth=1, antialiased=False)
        (self.cm,) = self.ax["ax_3d"].plot([], [], [], color=color, marker=".")
        (self.traj,) = self.ax["ax_3d"].plot([], [], [], color=color, marker=".")

        self.trajectory = {
            "t": [], "x": [], "y": [], "z": [], "psi": [],
        }

        (self.x_bar,) = self.ax["ax_error_x"].plot([], [], color=color, label=f"id: {self.uav.id}")
        (self.y_bar,) = self.ax["ax_error_y"].plot([], [], color=color, label=f"id: {self.uav.id}")
        (self.z_bar,) = self.ax["ax_error_z"].plot([], [], color=color, label=f"id: {self.uav.id}")
        (self.psi_bar,) = self.ax["ax_error_psi"].plot([], [], label=f"id: {self.uav.id}")

        l = self.uav.r * 2  # Assuming l is proportional to UAV radius
        self.points = np.array([
            [-l, 0, 0], [l, 0, 0], [0, -l, 0], [0, l, 0], [0, 0, 0],  # cm
            [-l, 0, 0], [l, 0, 0], [0, -l, 0], [0, l, 0],
        ]).T

    def update(self, t, done=False):
        R = self.uav.rotation_matrix()

        body = np.dot(R, self.points)
        body[0, :] += self.uav.state[0]
        body[1, :] += self.uav.state[1]
        body[2, :] += self.uav.state[2]

        self.l1.set_data(body[0, 0:2], body[1, 0:2])
        self.l1.set_3d_properties(body[2, 0:2])
        self.l2.set_data(body[0, 2:4], body[1, 2:4])
        self.l2.set_3d_properties(body[2, 2:4])
        self.cm.set_data(body[0, 4:5], body[1, 4:5])
        self.cm.set_3d_properties(body[2, 4:5])

        if self.pad_sprite:
            self.pad_sprite.remove()

        self.trajectory["t"].append(t)
        self.trajectory["x"].append(self.uav.state[0])
        self.trajectory["y"].append(self.uav.state[1])
        self.trajectory["z"].append(self.uav.state[2])
        self.trajectory["psi"].append(self.uav.state[8])

        self.ax["ax_error_x"].set_xlim(left=max(0, t - self.t_lim), right=t + self.t_lim)
        self.x_bar.set_data(self.trajectory["t"], self.trajectory["x"])
        self.ax["ax_error_y"].set_xlim(left=max(0, t - self.t_lim), right=t + self.t_lim)
        self.y_bar.set_data(self.trajectory["t"], self.trajectory["y"])
        self.ax["ax_error_z"].set_xlim(left=max(0, t - self.t_lim), right=t + self.t_lim)
        self.z_bar.set_data(self.trajectory["t"], self.trajectory["z"])
        self.ax["ax_error_psi"].set_xlim(left=max(0, t - self.t_lim), right=t + self.t_lim)
        self.psi_bar.set_data(self.trajectory["t"], self.trajectory["psi"])

        if done:
            self.traj.set_data(self.trajectory["x"], self.trajectory["y"])
            self.traj.set_3d_properties(self.trajectory["z"])

class Gui:
    def __init__(self, uavs={}, target=None, obstacles=[], vertiport=None, grid_points=[], max_x=5, max_y=5, max_z=5, fig=None):
        self.uavs = uavs
        self.fig = fig
        self.target = target
        self.obstacles = obstacles
        self.vertiport = vertiport
        self.grid_points = grid_points
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.cmap = plt.get_cmap("tab10")

        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 6))
            gs0 = self.fig.add_gridspec(1, 2)
            gs00 = gs0[0].subgridspec(1, 1)
            gs01 = gs0[1].subgridspec(4, 1)
            self.ax = {}
            self.ax["ax_3d"] = self.fig.add_subplot(gs00[0], projection="3d")
            self.ax["ax_error_x"] = self.fig.add_subplot(gs01[0])
            self.ax["ax_error_x"].set_ylim([-max_x, max_x])
            self.ax["ax_error_x"].set_ylabel("x (m)")
            self.ax["ax_error_y"] = self.fig.add_subplot(gs01[1])
            self.ax["ax_error_y"].set_ylim([-max_y, max_y])
            self.ax["ax_error_y"].set_ylabel("y (m)")
            self.ax["ax_error_z"] = self.fig.add_subplot(gs01[2])
            self.ax["ax_error_z"].set_ylim([0, max_z])
            self.ax["ax_error_z"].set_ylabel("z (m)")
            self.ax["ax_error_psi"] = self.fig.add_subplot(gs01[3])
            self.ax["ax_error_psi"].set_ylim([-np.pi, np.pi])
            self.ax["ax_error_psi"].set_ylabel(r"$\psi$ (rad)")

        self.ax["ax_3d"].set_xlim3d([-self.max_x, self.max_x])
        self.ax["ax_3d"].set_ylim3d([-self.max_y, self.max_y])
        self.ax["ax_3d"].set_zlim3d([0, self.max_z])

        self.ax["ax_3d"].set_xlabel("X (m)")
        self.ax["ax_3d"].set_ylabel("Y (m)")
        self.ax["ax_3d"].set_zlabel("Z (m)")

        self.ax["ax_3d"].set_title("Multi-UAV Simulation")

        self.time_display = self.ax["ax_3d"].text2D(0.0, 0.95, "", color="red", transform=self.ax["ax_3d"].transAxes)
        self.init_entities()

        # For stopping simulation with the esc key
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_routine)

        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def init_entities(self):
        c_idx = 0
        self.sprites = []
        if self.target:
            target_sprite = TargetSprite(self.ax, self.target, color=self.cmap(c_idx))
            self.sprites.append(target_sprite)

        for uav in self.uavs.values():
            c_idx += 1
            uav_sprite = UavSprite(self.ax, uav, color=self.cmap(c_idx))
            self.sprites.append(uav_sprite)

        for obstacle in self.obstacles:
            obs_sprite = ObstacleSprite(self.ax, obstacle)
            self.sprites.append(obs_sprite)

    def update(self, time_elapsed, done=False):
        self.fig.canvas.restore_region(self.background)
        self.time_display.set_text(f"Sim time = {time_elapsed:.2f} s")

        for sprite in self.sprites:
            sprite.update(time_elapsed, done)

        self.fig.canvas.blit(self.fig.bbox)
        # Only plot legends if UAV or target exists
        for key, ax in self.ax.items():
            if key == "ax_3d":
                continue

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        plt.pause(0.01)
        return self.fig

    def keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == "x":
            y = list(self.ax["ax_3d"].get_ylim3d())
            y[0] += 0.2
            y[1] += 0.2
            self.ax["ax_3d"].set_ylim3d(y)
        elif event.key == "w":
            y = list(self.ax["ax_3d"].get_ylim3d())
            y[0] -= 0.2
            y[1] -= 0.2
            self.ax["ax_3d"].set_ylim3d(y)
        elif event.key == "d":
            x = list(self.ax["ax_3d"].get_xlim3d())
            x[0] += 0.2
            x[1] += 0.2
            self.ax["ax_3d"].set_xlim3d(x)
        elif event.key == "a":
            x = list(self.ax["ax_3d"].get_xlim3d())
            x[0] -= 0.2
            x[1] -= 0.2
            self.ax["ax_3d"].set_xlim3d(x)
        elif event.key == "escape":
            exit(0)

    def __del__(self):
        plt.close(self.fig)

    def close(self):
        plt.close(self.fig)
        self.fig = None
