import numpy as np
import pure_pursuit as pp
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["axes.titlesize"] = 10 # for big titles
cmap = matplotlib.colormaps['viridis']

n_points = 100
dol_min = 1.1
dol_max = 20
eps = 0.00001 # numerical stability in arcsin

def custom_colorbar(min, max, **kwargs):
    norm = matplotlib.colors.Normalize(min, max)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), **kwargs)

def plot_phi_cap(mu=1, n_lines=100):

    lod = np.linspace(1/dol_max, 1/dol_min, n_lines)
    dol = 1/lod
    th_max = np.arcsin(2/dol - 1)
    theta = np.linspace(-np.pi/2, th_max-eps, n_points)

    plt.title(f"Pursuer Heading at Capture against Evader Heading for $\mu={mu}$")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi_c$", rotation=0)
    for i in range(n_lines):
        phi_cap = pp.phi_cap_1(theta[:,i], dol[i], mu)
        plt.plot(theta[:,i], phi_cap, color=cmap(i/n_lines))

    custom_colorbar(1/dol_max, 1/dol_min, label="l/d")

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

def plot_t_cap(mu, n_lines=100):
    lod = np.linspace(1/dol_max, 1/dol_min, n_lines)
    th_max = np.arcsin(2*lod - 1)
    th_min = np.clip(-np.pi - th_max, -np.pi/2, np.pi/2)
    theta = np.linspace(th_min+eps, th_max-eps, n_points)
    
    plt.title(f"Distance Normalized Time to Capture against Evader Heading for $\mu={mu}$")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\frac{t_c}{d}$", rotation=0)

    for i in range(n_lines):
        t_cap = pp.t_cap_1(theta[:,i], lod[i], mu)
        # if any(t_cap < 0):
        #     continue # don't ask
        plt.plot(theta[:,i], t_cap, color=cmap(i/n_lines))

    custom_colorbar(1/dol_max, 1/dol_min, label="l/d")
    

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

    for mu in [0.1, 0.5, 0.75, 0.99, 1, 1.01, 1.25, 2, 3]:
        _, phi, _ = plot_r(mu, 30)
        plt.savefig(f"{args.output}/r_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()
    for mu in [1, 2]:
        plot_phi_cap(mu, 30)
        plt.savefig(f"{args.output}/phi_cap_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()
        plot_t_cap(mu, 30)
        plt.savefig(f"{args.output}/t_cap_mu_{mu}.{args.format}")
        plt.cla()
        plt.clf()
