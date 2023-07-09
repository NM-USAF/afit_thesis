import pygame
import numpy as np
from simulation_core import *
    
    
class MouseController():
    def __init__(self, scenario:EngagementModel):
        self.scenario = scenario
        self.active_pursuer = None
        self.moving_pursuer = False

    def handle_mouse_down(self, mouse_x, mouse_y):
        # pick it up
        if self.active_pursuer:
            self.moving_pursuer = True

    def handle_mouse_up(self, mouse_x, mouse_y):
        self.moving_pursuer = False

    def handle_mouse_move(self, mouse_x, mouse_y):
        if self.moving_pursuer:
            self.active_pursuer.x = mouse_x
            self.active_pursuer.y = mouse_y
        else:
            self.active_pursuer = self.scenario.get_nearest_pursuer(
                mouse_x, mouse_y, 0.3
            )

    def handle_mouse_scroll(self, mouse_x, mouse_y, amount):
        self.active_pursuer.l += amount
        if self.active_pursuer.l < 0:
            self.active_pursuer.l = 0

    def increase_capture_radius(self, amount):
        set_to = self.scenario.pursuers[0].l + amount
        for p in self.scenario.pursuers:
            p.l = set_to

    def add_pursuer_at(self, mouse_x, mouse_y):
        new_pursuer = PursuerWorldState(
            mouse_x,
            mouse_y,
            1.9,
            1.0
        )
        self.scenario.add_pursuer(new_pursuer)

    def simulate(self):
        self.scenario.simulate(10)


PURSUER_COLOR = "red"
EVADER_COLOR = "blue"
WORLD_AGENT_RADIUS = 0.2
class PyGameView():
    def __init__(
            self, 
            scenario:EngagementModel, 
            controller, 
            canvas_size=300, 
            world_size=20
        ):
        self.scenario = scenario
        self.controller = controller
        self.canvas_size = canvas_size

        self.world_canvas_ratio = canvas_size / world_size
        self.screen = pygame.display.set_mode((canvas_size, canvas_size))

        self.show_simulation = False
        self.simulation_time = 0


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


    def draw_path(self, world_path, color):
        path = [
            self.world_to_canvas(p.x, p.y)
            for p in world_path
        ]

        pygame.draw.lines(
            self.screen,
            color,
            False,
            path
        )


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
            
        if self.show_simulation:
            times = np.linspace(0, 5)
            e_path, p_paths = self.scenario.get_state_for_times(times)

            for i in range(len(p_paths[0])):
                path = [ p_paths[t][i] for t in range(len(p_paths)) ]
                self.draw_path(path, PURSUER_COLOR)

            self.draw_path(e_path, EVADER_COLOR)

            e, ps = self.scenario.get_state_for_time(self.simulation_time)

            self.render_evader(e)
            for p in ps:
                self.render_pursuer(p, e)

        else:
            e = scenario.evader
            for p in scenario.world_pursuers:
                self.render_pursuer(p, e)

            self.render_evader(e)


    def step_game(self):
        distance_margin = self.scenario.optimize_evader_heading()

        if self.show_simulation:            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                self.simulation_time += 1/30
            elif keys[pygame.K_LEFT]:
                self.simulation_time -= 1/30

            self.simulation_time = np.clip(self.simulation_time, 0, 10)
        
        self.success = distance_margin > 0


    def handle_event(self, pygame_event):
        x, y = self.canvas_to_world(*pygame.mouse.get_pos())

        if self.show_simulation:
            if pygame_event.type == pygame.KEYDOWN:
                if pygame_event.key == pygame.K_SPACE:
                    self.show_simulation = False


        else:
            if pygame_event.type == pygame.MOUSEMOTION:
                self.controller.handle_mouse_move(x, y)
            elif pygame_event.type == pygame.MOUSEWHEEL:
                amount = pygame_event.y * 0.1
                self.controller.increase_capture_radius(amount)
            elif pygame_event.type == pygame.MOUSEBUTTONDOWN:
                if not pygame.mouse.get_pressed()[0]:
                    # not left mouse button
                    return 
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # control is pressed, add a new pursuer
                    self.controller.add_pursuer_at(x, y)
                else:
                    self.controller.handle_mouse_down(x, y)
            elif pygame_event.type == pygame.KEYDOWN:
                if pygame_event.key == pygame.K_SPACE:
                    self.controller.simulate()
                    self.show_simulation = True
            elif pygame_event.type == pygame.MOUSEBUTTONUP:
                self.controller.handle_mouse_up(x, y)


if __name__ == "__main__":

    np.seterr(all="raise")

    sc = PurePursuitScenario()
    sc.evader.v = 2

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