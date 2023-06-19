from collections import namedtuple
from dataclasses import dataclass
import numpy as np
import pygame
import pure_pursuit as pp
import utilities

# velocity components are not needed because a pure pursuer always travels
# towards the evader's position
@dataclass
class PursuerWorldState:
    x: float
    y: float
    v: float
    l: float

# evader theta is in the world frame
# theta = 0 is in the positive x direction
@dataclass
class EvaderWorldState:
    x: float
    y: float
    v: float
    theta: float

# theta necessary here because it will be different for each pursuer
# theta must be between -pi/2 and pi/2, so parity exists to indicate that
# theta in the original pursuer frame lied outside that range - the evader's 
# trajectory is to the left of the pursuer (parity=-1) instead of the right
# (parity = 1)
PursuerEngagementState = namedtuple(
    "PursuerEngagementState",
    ["r", "phi", "mu", "l", "theta", "parity"]
)


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


class Engagement2v1():
    """
    2 pursuers trying to catch 1 evader
    """

    def __init__(
        self, 
        e:EvaderWorldState, 
        pw_1:PursuerWorldState,
        pw_2:PursuerWorldState
    ):
        # evader saved only for conversion purposes
        self.evader = e
        pe_1 = world_to_engagement(e, pw_1)
        pe_2 = world_to_engagement(e, pw_2)

        # determine "left" and "right" pursuers
        # 2d cross product (x1*y2 - x2*y1) = ||v1||*||v2||*sin(theta)
        # where 'theta' is the angle between the vectors.
        # So 'left' = v2 and 'right' = v1 -> v1 x v2 > 0
        delta_x_p1 = pw_1.x - e.x
        delta_y_p1 = pw_1.y - e.y
        delta_x_p2 = pw_2.x - e.x
        delta_y_p2 = pw_2.y - e.y

        cross = (delta_x_p1 * delta_y_p2) - (delta_x_p2 * delta_y_p1)

        # also determine the angle between pursuers using the same math while we
        # have all the variables lying around
        dot = delta_x_p1 * delta_x_p2 + delta_y_p1 * delta_y_p2

        if cross >= 0:
            self.p_left = pe_2
            self.p_right = pe_1
        elif cross < 0:
            self.p_left = pe_1
            self.p_right = pe_2

        self.angle_between = np.arccos(dot / self.p_left.r / self.p_right.r)



    def optimal_evader_heading(self):
        theta_l = pp.optimal_evader_heading(
            self.p_left.r / self.p_right.r,
            self.p_left.l / self.p_left.r, 
            self.p_right.l / self.p_right.r,
            self.p_left.mu, self.p_right.mu,
            self.angle_between,
            n_iters=10
        )

        min_d = pp.r_min(theta_l, self.p_left.mu) * self.p_left.r - self.p_left.l

        p_left_w = engagement_to_world(self.evader, self.p_left)
        delta_x = p_left_w.x - self.evader.x
        delta_y = p_left_w.y - self.evader.y
        angle = np.arctan2(delta_y, delta_x)

        return angle - theta_l - np.pi/2, min_d


class PurePursuitScenario():
    """
    N pursuers against one evader
    Evader is at (0, 0)
    """

    def __init__(self, world_size=10):
        self.pursuers = []
        self.world_size = world_size

        self.evader = EvaderWorldState(
            0, 0, 3, 0
        )

    def add_pursuer(self, p:PursuerWorldState):
        self.add_pursuers([p])

    def add_pursuers(self, ps):
        self.pursuers += ps

    def get_nearest_index(self, wx, wy, min_dist=None):
        xs = np.array([ wx - p.x for p in self.pursuers ])
        ys = np.array([ wy - p.y for p in self.pursuers ])
        dists = np.hypot(xs, ys)

        if min_dist and np.min(dists) < min_dist:
            return np.argmin(np.hypot(xs, ys))
        
        return -1
    

    def optimal_evader_heading(self):
        xs = np.array([ p.x - self.evader.x for p in self.pursuers ])
        ys = np.array([ p.y - self.evader.y for p in self.pursuers ])
        angles = np.arctan2(ys, xs)
        indices = np.argsort(angles)
        indices = np.append(indices, indices[0])

        headings = []
        distances = []

        # adjacent pairs of pursuers
        for i_r, i_l in zip(indices[:-1], indices[1:]):
            eng = Engagement2v1(
                self.evader, 
                self.pursuers[i_r], 
                self.pursuers[i_l]
            )
            heading, distance = eng.optimal_evader_heading()
            headings.append(heading)
            distances.append(distance)

        max_i = np.nanargmax(distances)

        print("-----------")
        print(headings)
        print(distances)

        return headings[max_i], distances[max_i]
    

    def optimize_evader_heading(self):
        opt_heading, opt_dist = self.optimal_evader_heading()
        if np.isnan(opt_heading):
            self.evader.theta = 0
        else:
            self.evader.theta = opt_heading
        return opt_dist

    
class MouseController():
    def __init__(self, scenario:PurePursuitScenario):
        self.scenario = scenario
        self.active_pursuer_index = -1
        self.moving_pursuer = False

    def get_active_pursuer(self):
        return self.scenario.pursuers[self.active_pursuer_index]

    def handle_mouse_down(self, mouse_x, mouse_y):
        # pick it up
        if self.active_pursuer_index >= 0:
            self.moving_pursuer = True

    def handle_mouse_up(self, mouse_x, mouse_y):
        self.moving_pursuer = False

    def handle_mouse_move(self, mouse_x, mouse_y):
        if self.moving_pursuer:
            p = self.get_active_pursuer()
            p.x = mouse_x
            p.y = mouse_y
        else:
            idx = self.scenario.get_nearest_index(mouse_x, mouse_y, min_dist=0.3)
            self.active_pursuer_index = idx

    def handle_mouse_scroll(self, mouse_x, mouse_y, amount):
        p = self.get_active_pursuer()
        p.l += amount
        if p.l < 0:
            p.l = 0

    def add_pursuer_at(self, mouse_x, mouse_y):
        new_pursuer = PursuerWorldState(
            mouse_x,
            mouse_y,
            1,
            1.0
        )
        self.scenario.add_pursuer(new_pursuer)

PURSUER_COLOR = "red"
EVADER_COLOR = "blue"
WORLD_AGENT_RADIUS = 0.2
class PyGameView():
    def __init__(self, scenario, controller, canvas_size=300):
        self.scenario = scenario
        self.controller = controller
        self.canvas_size = canvas_size

        self.world_canvas_ratio = canvas_size / scenario.world_size
        self.screen = pygame.display.set_mode((canvas_size, canvas_size))


    def world_to_canvas(self, world_x, world_y):
        canvas_x = world_x * self.world_canvas_ratio + self.canvas_size / 2
        canvas_y = self.canvas_size / 2 - world_y * self.world_canvas_ratio
        return canvas_x, canvas_y


    def canvas_to_world(self, canvas_x, canvas_y):
        world_x = (canvas_x - self.canvas_size / 2) / self.world_canvas_ratio
        world_y = (self.canvas_size / 2 - canvas_y) / self.world_canvas_ratio
        return world_x, world_y


    def draw_vector(self, origin_x, origin_y, dir_x, dir_y, len, color, **kwargs):
        dir_len = np.hypot(dir_x, dir_y)
        v_x = dir_x / dir_len * len
        v_y = dir_y / dir_len * len

        end = self.world_to_canvas(origin_x + v_x, origin_y + v_y)
        pos = self.world_to_canvas(origin_x, origin_y)

        pygame.draw.line(self.screen, color, pos, end, **kwargs)


    def render_pursuer(self, p_world:PursuerWorldState, e_world:EvaderWorldState):
        pos = self.world_to_canvas(p_world.x, p_world.y)

        canvas_r = WORLD_AGENT_RADIUS * self.world_canvas_ratio
        canvas_l = p_world.l * self.world_canvas_ratio

        pygame.draw.circle(self.screen, PURSUER_COLOR, pos, canvas_r)
        pygame.draw.circle(self.screen, PURSUER_COLOR, pos, canvas_l, width=2)

        self.draw_vector(
            p_world.x, p_world.y, e_world.x - p_world.x, e_world.y - p_world.y,
            p_world.v, PURSUER_COLOR    
        )


    def render_evader(self, evader:EvaderWorldState):
        pos = self.world_to_canvas(evader.x, evader.y)

        pygame.draw.circle(
            self.screen, EVADER_COLOR, pos, 
            WORLD_AGENT_RADIUS * self.world_canvas_ratio
        )

        v_x, v_y = np.cos(evader.theta), np.sin(evader.theta)

        self.draw_vector(
            evader.x, evader.y, v_x, v_y, evader.v, EVADER_COLOR
        )


    def render_scenario(self, scenario:PurePursuitScenario=None):
        if not scenario:
            scenario = self.scenario
            
        e = scenario.evader
        for p in scenario.pursuers:
            self.render_pursuer(p, e)

        self.render_evader(e)


    def step_game(self):
        distance_margin = self.scenario.optimize_evader_heading()
        self.success = distance_margin > 0

        
    def handle_event(self, pygame_event):
        x, y = self.canvas_to_world(*pygame.mouse.get_pos())
        if pygame_event.type == pygame.MOUSEMOTION:
            self.controller.handle_mouse_move(x, y)
        elif pygame_event.type == pygame.MOUSEWHEEL:
            amount = pygame_event.y * 0.1
            self.controller.handle_mouse_scroll(x, y, amount)
        elif pygame_event.type == pygame.MOUSEBUTTONDOWN:
            if not pygame.mouse.get_pressed()[0]:
                # not left mouse button
                return 
            if pygame.key.get_mods() & pygame.KMOD_CTRL:
                # control is pressed, add a new pursuer
                self.controller.add_pursuer_at(x, y)
            else:
                self.controller.handle_mouse_down(x, y)
        elif pygame_event.type == pygame.MOUSEBUTTONUP:
            self.controller.handle_mouse_up(x, y)


if __name__ == "__main__":

    sc = PurePursuitScenario(world_size=20)

    ct = MouseController(sc)
    ct.add_pursuer_at(5, 2)
    ct.add_pursuer_at(5, -3)

    pgv = PyGameView(sc, ct, canvas_size=500)

    clock = pygame.time.Clock()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                pgv.handle_event(event)

        pgv.screen.fill("white")

        pgv.render_scenario()

        pgv.step_game()
    
        pygame.display.flip()

        dt = clock.tick(30) / 1000

    pygame.quit()