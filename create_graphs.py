import numpy as np
import pure_pursuit as pp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patch
import utilities

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

    plt.title(f"Minimum pursuer-evader distance")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$min(\frac{r}{d})$")

    for i in range(n_lines):
        r_m = pp.r_min(theta, mu[i])
        plt.plot(theta, r_m, color=cmap(i/n_lines))

    plt.plot(theta, (1+np.sin(theta))/2, color="black", label=f"limits at $\mu=1$ and $\mu=\inf$")
    plt.plot(theta, np.where(theta < 0, np.cos(theta), 1), color="black")

    m_l, b_l, m_u, b_u = pp.r_min_range_params()

    plt.plot(theta, m_l * theta + b_l)
    plt.plot(theta, m_u * theta + b_u)

    plt.legend()

    custom_colorbar(mu_min, mu_max, label=r"$\mu$")


# note: needs to be fixed
def plot_optimal_evader_heading(mu_min, mu_max, n_lines=100):
    mu = np.linspace(mu_min, mu_max, n_lines)
    dr_over_dl = np.linspace(0, 2.99, n_points)

    plt.title(r"Optimal evader heading for two identical pursuers with a gap of $\frac{\pi}{3}$ between them")
    plt.xlabel(r"Pursuer distance ratio $\frac{d_l}{d_r}$")
    plt.ylabel(r"Optimal constant evader heading from left pursuer $\theta_l$")

    dl = np.ones(len(dr_over_dl))
    dr = dr_over_dl
    angle_between = np.ones(len(dr_over_dl)) * np.pi/3
    for i in range(n_lines):
        th_l = pp.optimal_evader_heading(dl/dr, 1, 1, angle_between, mu[i], mu[i])
        
        plt.plot(dr_over_dl, th_l, color=cmap(i/n_lines))

    custom_colorbar(mu_min, mu_max, label=r"$\mu$")


def plot_example_multiple_pursuit_min_r(n_pursuers=5):
    headings = np.random.uniform(-np.pi, np.pi, n_pursuers)
    distances = np.random.uniform(1.5, 5, n_pursuers)
    lods = np.random.uniform(0.1, 0.7, n_pursuers)
    mus = np.random.uniform(1.1, 5, n_pursuers)

    theta_e = np.linspace(-np.pi, np.pi, 314)

    theta_p = utilities.wrap(theta_e[:,None] - headings - np.pi/2, np.pi)
    
    rod_min = pp.r_min(theta_p, mus)
    r_minus_l = (rod_min - lods) * distances

    ml, bl, mu, bu = pp.r_min_range_params()

    for i in range(n_pursuers):
        label = fr"$\mu={mus[i]:.2f}, d={distances[i]:.2f}, l={lods[i]*distances[i]:.2f}, \theta={headings[i]:.2f}$"
        for j in range(n_pursuers):
            if i == j:
                continue
            l, r, b, t = pp.optimal_evader_heading_region(
                distances[i], distances[j],
                lods[i], lods[j],
                headings[i] - headings[j]
            )
            l += headings[i] + np.pi/2
            r += headings[i] + np.pi/2
            # plt.vlines([l, r], b, t)
            # rect = patch.Rectangle(
            #     (l, b), 
            #     (r - l),
            #     (t - b),
            #     alpha=0.1
            # )
            # plt.gca().add_patch(rect)
            print(l, r, b, t)

        plt.plot(theta_e, r_minus_l[:,i], label=label)   
        t_p = utilities.fix_theta(theta_p[:,i])
        r_l_bound = ml*(t_p) + bl
        r_u_bound = mu*(t_p) + bu

        print(min(r_l_bound))

        max_t_p = np.arcsin(1/mus[i])
        where_valid = (
            (t_p > -np.pi/2) & (t_p < max_t_p) |
            ((t_p > 3/2*np.pi) & (t_p < utilities.mirror(max_t_p, np.pi/2)))
        )
        r_l_bound = r_l_bound[where_valid]
        r_u_bound = r_u_bound[where_valid]
        theta_e_less = theta_e[where_valid]

        # plt.scatter(theta_e_less, distances[i]*(r_l_bound - lods[i]), s=1)
        # plt.scatter(theta_e_less, distances[i]*(r_u_bound - lods[i]), s=1)

    plt.plot(theta_e, np.min(r_minus_l, axis=1), color="black", label=r"minimum capture margin")
    plt.title("Minimum pursuer-evader capture margin for multiple pursuers")
    plt.xlabel(r"Evader world frame heading $\theta_e$ in $[-\pi, \pi]$")
    plt.ylabel(r"Capture margin $r_min - l$")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


class PlotSaver():
    def __init__(self, output_path, format):
        self.output_path = output_path
        self.format = format

    def save(self, name, **kwargs):
        plt.savefig(
            f"{args.output}/{name}.{args.format}", 
            bbox_extra_artists=(plt.gca().get_children())    ,
            **kwargs
        )
        plt.cla()
        plt.clf()

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

    saver = PlotSaver(args.output, args.format)

    np.set_printoptions(precision=2)

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    n_lines=50

    for mu in [0.1, 0.5, 0.75, 0.99, 1, 1.01, 1.25, 1.5, 2, 3]:
        plot_r(mu, n_lines)
        saver.save(f"r_mu_{mu}")

    for mu in [1, 2]:
        plot_phi_cap(mu, n_lines)
        saver.save(f"phi_cap_mu_{mu}")

        plot_t_cap(mu, n_lines)
        saver.save(f"t_cap_mu_{mu}")

    plot_min_r(1, 10, n_lines*2)
    saver.save("r_min")

    plot_mu_capture_ratio(2, 3, 12)
    saver.save("poly_dist_cap")

    plot_capture_ratio(
        np.linspace(1, 4, 13),
        np.arange(3, 16)
    )
    saver.save("poly_dist_cap_3d")

    plot_optimal_evader_heading(1, 5, 20)
    saver.save("optimal_evader_heading")

    plot_example_multiple_pursuit_min_r(2)
    saver.save(
        "multiple_pursuit_min_dist",
        bbox_inches="tight"
    )
