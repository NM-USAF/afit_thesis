from dataclasses import dataclass, replace
import numpy as np
import pure_pursuit as pp
import utilities
from abc import ABC, abstractproperty, abstractmethod
from typing import List, Tuple

# velocity components are not needed because a pure pursuer always travels
# towards the evader's position
@dataclass
class PursuerWorldState:
    x: float
    y: float
    v: float
    l: float

    x_idx = 0
    y_idx = 1
    v_idx = 2
    l_idx = 3

    def to_numpy(self):
        return np.array([self.x, self.y, self.v, self.l])
    
    def from_numpy(nparr):
        return PursuerWorldState(
            nparr[PursuerWorldState.x_idx],
            nparr[PursuerWorldState.y_idx],
            nparr[PursuerWorldState.v_idx],
            nparr[PursuerWorldState.l_idx]
        )

    def __len__(self):
        return 4


# evader theta is in the world frame.
# theta = 0 is in the positive x direction in the world frame
@dataclass
class EvaderWorldState:
    x: float
    y: float
    v: float
    theta: float

    x_idx = 0
    y_idx = 1
    v_idx = 2
    l_idx = 3

    def to_numpy(self):
        return np.array([self.x, self.y, self.v, self.theta])
    
    def __len__(self):
        return 4


# theta necessary here because it will be different for each pursuer.
# theta must be between -pi/2 and pi/2, so parity exists to indicate whether
# theta in the original pursuer frame lied outside that range - parity=-1 means
# the evader's trajectory is to the left of the pursuer instead of the right
@dataclass
class PursuerEngagementState:
    r: float
    phi: float
    mu: float
    l: float
    theta: float
    parity: int

    r_idx = 0
    phi_idx = 1
    mu_idx = 2
    l_idx = 3
    theta_idx = 4
    parity_idx = 5

    def to_numpy(self):
        return np.array([
            self.r, self.phi, self.mu, self.l, self.theta, self.parity
        ])
    
    def from_numpy(nparr):
        return PursuerEngagementState(
            nparr[PursuerEngagementState.r_idx],
            nparr[PursuerEngagementState.phi_idx],
            nparr[PursuerEngagementState.mu_idx],
            nparr[PursuerEngagementState.l_idx],
            nparr[PursuerEngagementState.theta_idx],
            nparr[PursuerEngagementState.parity_idx],
        )

    def __len__(self):
        return 6


def world_to_engagement(e:EvaderWorldState, p:PursuerWorldState):
    """
    converts a world-frame state of a pursuer/evader pair into an initial
    engagement state for the pursuer
    """
    delta_x = p.x - e.x
    delta_y = p.y - e.y
    r = np.hypot(delta_x, delta_y)
    mu = e.v / p.v

    # because this is the start of the engagement, 
    # psi = phi - theta and phi(0) = pi/2 -> theta = pi/2 - psi
    psi = np.arctan2(-delta_y, -delta_x) - e.theta
    psi = utilities.wrap(psi, np.pi)

    # theta in [-pi/2, pi/2] -> psi in [0, pi]
    parity = np.sign(psi)
    psi = np.abs(psi)
    theta_p = np.pi/2 - psi

    return PursuerEngagementState(r, np.pi/2, mu, p.l, theta_p, parity)


def engagement_to_world(e:EvaderWorldState, p:PursuerEngagementState):
    v = e.v / p.mu
    
    psi = (p.phi - p.theta) * p.parity
    psi_world = psi + e.theta
    delta_x = -p.r * np.cos(psi_world)
    delta_y = -p.r * np.sin(psi_world)
    return PursuerWorldState(e.x + delta_x, e.y + delta_y, v, p.l)


class EngagementModel(ABC):
    @abstractproperty
    def evader(self) -> EvaderWorldState:
        pass


    @abstractproperty
    def world_pursuers(self) -> List[PursuerWorldState]:
        pass


    @abstractmethod
    def simulate(self, t_max:float):
        pass


    @abstractmethod
    def get_state_for_time(self, time:float) -> Tuple[EvaderWorldState, List[PursuerWorldState]]:
        pass


    def get_state_for_times(self, times:float):
        return list(zip(*[ self.get_state_for_time(t) for t in times ]))


    def get_nearest_pursuer(self, world_x, world_y, min_dist=None):
        pursuers = self.world_pursuers
        xs = np.array([ world_x - p.x for p in pursuers ])
        ys = np.array([ world_y - p.y for p in pursuers ])
        dists = np.hypot(xs, ys)

        if min_dist and np.min(dists) > min_dist:
            return None
        
        return self.world_pursuers[np.argmin(np.hypot(xs, ys))]
        

class Engagement2v1():
    """
    2 pursuers trying to catch 1 evader
    """

    def __init__(
        self, 
        e:EvaderWorldState, 
        pw_l:PursuerWorldState,
        pw_r:PursuerWorldState
    ):
        # evader saved only for conversion purposes
        self._evader = e
        self.p_left = world_to_engagement(e, pw_l)
        self.p_right = world_to_engagement(e, pw_r)

        # determine "left" and "right" pursuers
        # 2d cross product (x1*y2 - x2*y1) = ||v1||*||v2||*sin(theta)
        # where 'theta' is the angle between the vectors.
        # So 'left' = v2 and 'right' = v1 -> v1 x v2 > 0
        delta_x_pl = pw_l.x - e.x
        delta_y_pl = pw_l.y - e.y
        delta_x_pr = pw_r.x - e.x
        delta_y_pr = pw_r.y - e.y

        # also determine the angle between pursuers using the same math while we
        # have all the variables lying around
        dot = delta_x_pl * delta_x_pr + delta_y_pl * delta_y_pr
        # extra help to make sure arccos works
        dot_norm = np.clip(dot / self.p_left.r / self.p_right.r, -1, 1)
        self.angle_between = np.arccos(dot_norm)

        # the angle from right to left should be positive (counter-clockwise)
        cross = (delta_x_pr * delta_y_pl) - (delta_x_pl * delta_y_pr)
        if cross > 0:
            # self.angle_between = 2*np.pi - self.angle_between
            self.p_left, self.p_right = self.p_right, self.p_left

    def optimal_evader_heading(self):
        theta_l = pp.optimal_evader_heading(
            self.p_left.r / self.p_right.r,
            self.p_left.l / self.p_left.r, 
            self.p_right.l / self.p_right.r,
            self.p_left.mu, self.p_right.mu,
            self.angle_between,
            method="scipy"
        )

        min_d = pp.r_min(theta_l, self.p_left.mu) * self.p_left.r - self.p_left.l

        p_left_w = engagement_to_world(self.evader, self.p_left)
        delta_x = p_left_w.x - self.evader.x
        delta_y = p_left_w.y - self.evader.y
        angle = np.arctan2(delta_y, delta_x)

        return angle - theta_l - np.pi/2, min_d
    
    @property
    def evader(self) -> EvaderWorldState:
        return self._evader

    @property
    def world_pursuers(self) -> List[PursuerWorldState]:
        return [
            engagement_to_world(self.evader, p)
            for p in [self.p_left, self.p_right]
        ]


class PurePursuitScenario(EngagementModel):
    """
    N pursuers against one evader
    Evader is at (0, 0)
    """

    def __init__(self):
        self.pursuers = []

        self.simulated_pursuer_states = []
        self.times = []

        self._evader = EvaderWorldState(
            0, 0, 3, 0
        )

    @property
    def evader(self):
        return self._evader

    def add_pursuer(self, p:PursuerWorldState):
        self.add_pursuers([p])

    def add_pursuers(self, ps):
        self.pursuers += ps
    

    def optimal_evader_heading(self):
        xs = np.array([ p.x - self.evader.x for p in self.world_pursuers ])
        ys = np.array([ p.y - self.evader.y for p in self.world_pursuers ])
        angles = np.arctan2(ys, xs)
        indices = np.argsort(angles)
        indices = np.append(indices, indices[0])

        # headings = []
        # distances = []

        # adjacent pairs of world_pursuers
        # for i_r, i_l in zip(indices[:-1], indices[1:]):
        #     eng = Engagement2v1(
        #         self.evader, 
        #         self.world_pursuers[i_l], 
        #         self.world_pursuers[i_r]
        #     )
        #     heading, distance = eng.optimal_evader_heading()
        #     headings.append(heading)
        #     distances.append(distance)

        # max_i = np.nanargmax(distances)

        # print(distances)

        # return headings[max_i], distances[max_i]

        eng = Engagement2v1(
            self.evader, self.world_pursuers[0], self.world_pursuers[1]
        )

        heading, distance = eng.optimal_evader_heading()
        return heading, distance
    
    
    def simulate(self, t_max):
        n = 1000
        dt = t_max / n
        self.times = np.linspace(0, t_max, n)
        engagement_pursuers = [
            world_to_engagement(self.evader, p)
            for p in  self.world_pursuers
        ]
        engagement_thetas = np.array([ e.theta for e in engagement_pursuers ])
        engagement_mus = np.array([ e.mu for e in engagement_pursuers ])
        engagement_ds = np.array([ e.r for e in engagement_pursuers ])

        # necessary because phi_dot actually returns (dphi/dt) / vp
        engagement_vs = np.array([ e.v for e in self.world_pursuers ])

        # create the objects to get filled out by simulation
        initial_pursuer_states = np.array([
            e.to_numpy() for e in engagement_pursuers
        ])

        self.simulated_pursuer_states = np.repeat(initial_pursuer_states[None], n, axis=0)

        # numerical integration: trapezoid method
        for i in range(1, n):
            pursuers_last = self.simulated_pursuer_states[i-1]

            phi_last = pursuers_last[:,PursuerEngagementState.phi_idx]
            r_last = pursuers_last[:,PursuerEngagementState.r_idx]
            dphi_dt = pp.phi_dot(r_last, phi_last, engagement_thetas, engagement_mus) * engagement_vs

            phi_next = phi_last + dphi_dt * dt
            phi_next = np.clip(phi_next, engagement_thetas, np.pi/2)
            r_next = pp.r(phi_next, engagement_thetas, engagement_mus) * engagement_ds
            dphi_dt_next = pp.phi_dot(r_next, phi_next, engagement_thetas, engagement_mus) * engagement_vs

            phi = phi_last + (dphi_dt + dphi_dt_next) / 2 * dt
            phi = np.clip(phi, engagement_thetas, np.pi/2)
            r = pp.r(phi, engagement_thetas, engagement_mus) * engagement_ds

            self.simulated_pursuer_states[i,:,PursuerEngagementState.phi_idx] = phi
            self.simulated_pursuer_states[i,:,PursuerEngagementState.r_idx] = r


    def evader_at_time(self, t):
        d = self.evader.v * t
        ct = np.cos(self.evader.theta)
        st = np.sin(self.evader.theta)

        return EvaderWorldState(
            self.evader.x + d * ct,
            self.evader.y + d * st,
            self.evader.v,
            self.evader.theta
        )


    def get_state_for_time(self, time: float) -> Tuple[EvaderWorldState, List[PursuerWorldState]]:
        idx = min(
            np.searchsorted(self.times, time), 
            len(self.simulated_pursuer_states)-1
        )

        evader = self.evader_at_time(time)

        pursuers = [
            engagement_to_world(
                evader, 
                PursuerEngagementState(*a)
            )
            for a in self.simulated_pursuer_states[idx]
        ]

        return (evader, pursuers)
    

    @property
    def world_pursuers(self):
        return self.pursuers
    

    def optimize_evader_heading(self):
        opt_heading, opt_dist = self.optimal_evader_heading()
        if np.isnan(opt_heading):
            self.evader.theta = 0
        else:
            self.evader.theta = opt_heading
        return opt_dist
