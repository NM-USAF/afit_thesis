import numpy as np
import pure_pursuit as pp
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["axes.titlesize"] = 10 # for big titles
cmap = matplotlib.colormaps['viridis']

n_points = 100
lod_min = 0.01
lod_max = 0.99
eps = 0.0000001 # numerical stability in arcsin and tan

def custom_colorbar(min, max, **kwargs):
    norm = matplotlib.colors.Normalize(min, max)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), **kwargs)


def plot_r(mu, n_lines=100):
    theta = np.linspace(-np.pi/2+eps, np.pi/2-eps, n_lines)

    phi = np.linspace(theta, np.pi/2-eps, n_points)

    r = pp.r(phi, theta, mu)

    plt.title(f"Distance Ratio against Pursuer Heading for $\mu={mu}$")
    plt.xlabel(r"heading difference $\psi = \phi - \theta$")
    plt.ylabel(r"$\frac{r}{d}$", rotation=0)

    for i in range(n_lines):
        plt.plot(phi[:,i] - theta[i], r[:,i], color=cmap(i/n_lines))

    custom_colorbar(-np.pi/2, np.pi/2, label=r"evader heading $\theta$")

    return theta, phi, r


def plot_phi_cap(mu=1, n_lines=100):
    lod = np.linspace(lod_min, lod_max, n_lines)
    th_max = np.arcsin(2*lod - 1)
    theta = np.linspace(-np.pi/2, th_max-eps, n_points)

    plt.title(f"Pursuer Heading at Capture against Evader Heading for $\mu={mu}$")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi_c$", rotation=0)
    for i in range(n_lines):
        phi_cap = pp.phi_cap_1(theta[:,i], lod[i], mu)
        plt.plot(theta[:,i], phi_cap, color=cmap(i/n_lines))

    custom_colorbar(lod_min, lod_max, label="l/d")


def plot_t_cap(mu, n_lines=100):
    lod = np.linspace(lod_min, lod_max, n_lines)
    th_max = np.arcsin(2*lod - 1)
    
    # t_cap needs extra epsilon at the low end for a point near tan(pi/2)
    theta = np.linspace(-np.pi/2+0.1, th_max-eps, n_points)
    
    plt.title(f"Distance Normalized Time to Capture against Evader Heading for $\mu={mu}$")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\frac{t_c}{d}$", rotation=0)

    for i in range(n_lines):
        t_cap = pp.t_cap_1(theta[:,i], lod[i], mu)
        plt.plot(theta[:,i], t_cap, color=cmap(i/n_lines))

    custom_colorbar(lod_min, lod_max, label="l/d")


def plot_mu_capture_ratio(mu, n_start, n_end):
    ns = np.arange(n_start, n_end+1)

    crits_d = pp.polygon_formation_capture_ratio_d(mu, ns)
    crits_a = pp.polygon_formation_capture_ratio_a(mu, ns)

    plt.title(f"Distance ratio at which capture occurs in a regular polygon formation for $\mu={mu}$")
    plt.xlabel("Number of pursuers")
    plt.ylabel("Ratio (unitless)")

    plt.scatter(ns, crits_d, label=r"Threshold of $\frac{l}{d}$")
    plt.scatter(ns, crits_a, label=r"Threshold of $\frac{l}{a}$")

    plt.legend()


def plot_capture_ratio(mus, ns):
    ax = plt.axes(projection="3d")

    n_pts = len(mus) * len(ns)
    xs = np.zeros(n_pts)
    ys = np.zeros(n_pts)
    zs = np.zeros(n_pts)

    surf = np.zeros((len(mus), len(ns)))

    xv, yv = np.meshgrid(mus, ns, indexing='ij')
    zv = pp.polygon_formation_capture_ratio_d(xv, yv)
    for i in range(len(mus)):
        for j in range(len(ns)):
            idx = i * len(ns) + j
            mu = xv[i,j]
            n = yv[i,j]
            xs[idx] = mu
            ys[idx] = n
            zs[idx] = pp.polygon_formation_capture_ratio_d(mu, n)
            surf[i,j] = zs[idx]

    # ax.scatter(ys, xs, zs)
    ax.plot_surface(yv, xv, zv, cmap="viridis")
    ax.set_xlabel("Number of pursuers")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"Distance ratio $\frac{l}{d}$ required for capture")
    

def plot_min_r(mu_min, mu_max, n_lines=100):
    theta = np.linspace(-np.pi/2, np.pi/2, n_points)
    mu = np.linspace(mu_min, mu_max, n_lines)

    plt.title(f"Minimum pursuer-Evader Distance")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$min(\frac{r}{d})$")

    for i in range(n_lines):
        r_m = pp.r_min(theta, mu[i])
        plt.plot(theta, r_m, color=cmap(i/n_lines))

    custom_colorbar(mu_min, mu_max, label=r"$\mu$")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--format", 
        default="png", 
        help="plot image output format"
    )
    parser.add_argument(
        "-o", "--output",
        default="plots",
        help="output directory"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    n_lines=50

    for mu in [0.1, 0.5, 0.75, 0.99, 1, 1.01, 1.25, 1.5, 2, 3]:
        plot_r(mu, n_lines)
        plt.savefig(f"{args.output}/r_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()

    for mu in [1, 2]:
        plot_phi_cap(mu, n_lines)
        plt.savefig(f"{args.output}/phi_cap_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()

        plot_t_cap(mu, n_lines)
        plt.savefig(f"{args.output}/t_cap_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()

    plot_min_r(1, 4, n_lines)
    plt.savefig(f"{args.output}/r_min.{args.format}")
    plt.cla()
    plt.clf()

    plot_mu_capture_ratio(2, 3, 12)
    plt.savefig(f"{args.output}/poly_dist_cap.{args.format}")
    plt.cla()
    plt.clf()

    plot_capture_ratio(
        np.linspace(1, 4, 13),
        np.arange(3, 16)
    )
    plt.savefig(f"{args.output}/poly_dist_cap_3d.{args.format}")
    plt.cla()
    plt.clf()

