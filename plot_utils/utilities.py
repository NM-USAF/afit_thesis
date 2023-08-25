import matplotlib.pyplot as plt

def line_segment(ax:plt.Axes, p1, p2, color="black", **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, **kwargs)
    