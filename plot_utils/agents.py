import numpy as np
import pure_pursuit as pp
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import pure_pursuit.utilities
from dataclasses import dataclass

ENTITY_KWARGS = {"color": "black"}
ANNOTATE_KWARGS = {"textcoords":"offset pixels"}
ARROW_SCALE = 0.25

TEXT_HEIGHT = 7
TEXT_WIDTH = 14
def xytext_from_angle(angle, default_offset=np.array([5, 5])):
    """
    returns an x/y text offset that will not overlap with an arrow
    pointing from the text origin towards `angle`
    """
    x = np.cos(angle)
    y = np.sin(angle)

    x_offset, y_offset = default_offset
    if y > 0:
        y_offset = -default_offset[1] - TEXT_HEIGHT
    if x > 0:
        x_offset = -default_offset[0] - TEXT_WIDTH

    return (x_offset, y_offset)

@dataclass
class Entity:
    x: float
    y: float
    label: str
    theta: float = 0
    mu: float = 1
    l: float = 0

    def draw_location(self, ax: plt.Axes):
        ax.scatter(self.x, self.y, **ENTITY_KWARGS)
        xyoffset = xytext_from_angle(self.theta)
        ax.annotate(self.label, (self.x, self.y), xytext=xyoffset, **ANNOTATE_KWARGS)

    def velocity_vector(self):
        arrow_len =  1 / self.mu * ARROW_SCALE
        dy, dx = np.sin(self.theta) * arrow_len, np.cos(self.theta) * arrow_len
        return np.array([dx, dy])
    
    def location_vector(self):
        return np.array([self.x, self.y])
    
    def velocity_arrow_vector(self):
        return self.location_vector() + self.velocity_vector()

    def draw_velocity(self, ax: plt.Axes):
        dx, dy = self.velocity_vector()
        ax.arrow(
            self.x, self.y, dx, dy, 
            length_includes_head=True,
            head_width=0.02,
            color="black"
        )

    def point_to_location(self, x_other, y_other):
        self.theta = np.arctan2(y_other - self.y, x_other - self.x)

    def point_to_entity(self, other):
        self.point_to_location(other.x, other.y)

    def annotate_velocity(self, ax, label="$V$"):
        dx, dy = self.velocity_vector()
        xyoffset = xytext_from_angle(self.theta + np.pi)
        vx, vy = self.x + dx, self.y + dy
        ax.annotate(label, (vx, vy), xytext = xyoffset, **ANNOTATE_KWARGS)

    def draw_capture_radius(self, ax:plt.Axes):
        circle = pch.Circle((self.x, self.y), self.l, fill=False, color="black")
        ax.add_patch(circle)
