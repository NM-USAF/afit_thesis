import numpy as np
import pure_pursuit as pp
import matplotlib.pyplot as plt
import pure_pursuit.utilities

from plot_utils.agents import Entity
from plot_utils.angle_annotation import AngleAnnotation
import plot_utils.utilities as pu
from plot_utils.saver import PlotSaver

plt.rcParams["figure.figsize"] = (3, 2)

DEFAULT_EVADER = Entity(0, 0, "$E$")

def one_on_one(ax:plt.Axes):
    # pursuer
    p1 = Entity(0, -0.5, "$P$", mu=1.5)
    p1.point_to_entity(DEFAULT_EVADER)
    p1.draw_location(ax)
    p1.draw_velocity(ax)
    p1.annotate_velocity(ax, "$v_P$")

    # evader - let theta=pi/6
    DEFAULT_EVADER.theta = np.pi / 6
    DEFAULT_EVADER.draw_location(ax)
    DEFAULT_EVADER.draw_velocity(ax)
    DEFAULT_EVADER.annotate_velocity(ax, "$v_E$")

    # dashed line for distance
    # ax.plot([0, 0], [-0.5, 0], color="black", linestyle="dashed", zorder=0)
    pu.line_segment(ax, [0, 0], [0, -0.5], linestyle="dashed")
    d_loc = (p1.location_vector() + DEFAULT_EVADER.location_vector()) / 2
    ax.annotate("$d$", d_loc, (-20, 0), textcoords="offset pixels")


    # dotted line for angle
    # ax.plot([0, 0], [-0.5, 0.2], color="black", linestyle="dotted")
    pu.line_segment(ax, [0, 0], [0, 0.2], linestyle="dotted")
    evader_loc = DEFAULT_EVADER.location_vector()
    evader_vel = DEFAULT_EVADER.velocity_vector()
    AngleAnnotation(
        evader_loc, 
        evader_loc + evader_vel,
        evader_loc + np.array([0, 1]),
        text=r"$\psi$",
        ax=ax,
        textposition="outside",
        size=40
    )

    return ax

def two_on_one(ax):
    p1 = Entity(0.5, 0.5, "$P_1$")
    p2 = Entity(-0.5, 1, "$P_2$")
    e = Entity(0, 0, "$E$", -np.pi/3)

    p1.point_to_entity(e)
    p2.point_to_entity(e)

    for ent in [p1, p2, e]:
        ent.draw_location(ax)
        ent.draw_velocity(ax)

    # reference line
    pu.line_segment(ax, [-0.3, 0], [0.3, 0], linestyle="dotted")

    pu.line_segment(
        ax, 
        p1.location_vector(),
        e.location_vector(),
        linestyle="dotted"
    )

    pu.line_segment(
        ax,
        p2.location_vector(),
        e.location_vector(),
        linestyle="dotted"
    )

    # angles n stuff
    AngleAnnotation(
        e.location_vector(),
        [1, 0],
        p1.location_vector(),
        text=r"$\gamma_1$",
        ax=ax,
        textposition="outside"
    )
    AngleAnnotation(
        e.location_vector(),
        [1, 0],
        p2.location_vector(),
        text=r"$\gamma_2$",
        ax=ax,
        textposition="outside",
        size=60
    )
    AngleAnnotation(
        e.location_vector(),
        e.velocity_arrow_vector(),
        [1, 0], 
        text=r"$\theta_w$",
        ax=ax, 
        textposition="outside",
        size=40
    )

    return ax


def triangular_formation(ax):
    distance = 0.3
    gammas = np.array([1, 2, 3]) * (2*np.pi/3) + np.pi
    l = 1 / np.sqrt(3) * distance
    mu = 2
    pursuers = [
        Entity(np.cos(g) * distance, np.sin(g) * distance, f"$P_{i+1}$", mu=mu, l=l)
        for i, g in enumerate(gammas)
    ]

    for p in pursuers:
        p.point_to_entity(DEFAULT_EVADER)
        p.draw_location(ax)
        p.draw_velocity(ax)
        p.draw_capture_radius(ax)

    for i in range(len(pursuers)):
        p = pursuers[-i]
        p_next = pursuers[-i-1] # circular list moment
        pu.line_segment(
            ax, 
            p.location_vector(), 
            p_next.location_vector(),
            linestyle="dashed"
        )

    DEFAULT_EVADER.theta = 0
    DEFAULT_EVADER.draw_location(ax)
    DEFAULT_EVADER.draw_velocity(ax)

def new_axis():
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")
    return ax

if __name__ == "__main__":

    saver = PlotSaver("diagrams", "png")

    one_on_one(new_axis())
    saver.save(
        "one_on_one",
        latex=True
    )

    two_on_one(new_axis())
    saver.save("two_on_one", latex=True)
    
    triangular_formation(new_axis())
    saver.save("triangular_formation", latex=True)

    with open("diagrams.tex", "w") as f:
        f.write(saver.to_latex())