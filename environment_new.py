import random
import utils
import numpy as np
import pygame
from tensorflow import keras
from skimage import transform



class SnakeEnv:
    def __init__(self, settings):
        self.settings = settings
        self.width = (self.settings.grid_width + 2*self.settings.wall_pixels)*self.settings.pixel_size
        self.length = (self.settings.grid_length + 2*self.settings.wall_pixels)*self.settings.pixel_size
        self.wall_size = self.settings.wall_pixels*self.settings.pixel_size
        self.snake_size = self.settings.snake_pixels*self.settings.pixel_size

        self.action_space = self.settings.action_space
        self.states_space = np.zeros((self.width, self.length), dtype=np.uint8)

        self.score = 0

        self.firstcall = True
        self.env_clock = None
        self.screen = None
        self.score_font = None
        self.board_status = None

        self.possible_x = [i for i in range(self.wall_size, self.width-self.wall_size, self.snake_size)]
        self.possible_y = [i for i in range(self.wall_size, self.length-self.wall_size, self.snake_size)]

        self.states_space[:, 0:self.wall_size] = self.settings.wall_color.id
        self.states_space[:, -self.wall_size:] = self.settings.wall_color.id
        self.states_space[-self.wall_size:, :] = self.settings.wall_color.id
        self.states_space[0:self.wall_size, :] = self.settings.wall_color.id

    def init_grid(self):
        self.head_x = random.choice(self.possible_x)
        self.head_y = random.choice(self.possible_y)
        self.snake_head = [self.head_x, self.head_y]
        self.snake_list = [self.snake_head]

        second_x = 2 * self.wall_size if self.head_x == self.wall_size else self.head_x - self.snake_size
        second_y = self.head_y
        chunck = [second_x, second_y]
        self.snake_list.insert(0, chunck)

        self.foodx, self.foody = utils.add_food(self.possible_x, self.possible_y, self.snake_list)

    def reset(self):
        self.init_grid()
        self.board_status = np.copy(self.states_space)

        for i in range(self.snake_size):
            for j in range(self.snake_size):
                self.board_status[self.foody + i][self.foodx + j] = self.settings.f_color.id
                self.board_status[self.snake_head[1] + i][self.snake_head[0] + j] = self.settings.sh_color.id
                for b in self.snake_list[:-1]:
                    self.board_status[b[1] + i][b[0] + j] = self.settings.sb_color.id

        self.score = 0

        return self.board_status

    def get_state_map(self):
        self.board_status = np.copy(self.states_space)

        for i in range(self.snake_size):
            for j in range(self.snake_size):
                self.board_status[self.foody + i][self.foodx + j] = self.settings.f_color.id
                for b in self.snake_list[:-1]:
                    self.board_status[b[1] + i][b[0] + j] = self.settings.sb_color.id

        if self.wall_size <= self.snake_head[0] < self.width - self.wall_size and self.wall_size <= self.snake_head[
            1] < self.length - self.wall_size:
            for i in range(self.snake_size):
                for j in range(self.snake_size):
                    self.board_status[self.snake_head[1] + i][self.snake_head[0] + j] = self.settings.sh_color.id

        return self.board_status

    def step(self, action):
        terminal = False
        add_block = False
        if action == 3:
            x_change = -self.snake_size
            y_change = 0
        elif action == 1:
            x_change = self.snake_size
            y_change = 0
        elif action == 0:
            y_change = -self.snake_size
            x_change = 0
        elif action == 2:
            y_change = self.snake_size
            x_change = 0

        self.head_x += x_change
        self.head_y += y_change
        self.snake_head = [self.head_x, self.head_y]
        self.snake_list.append(self.snake_head)

        if self.head_x >= self.width - self.wall_size or self.head_x < self.wall_size or self.head_y >= self.length - self.wall_size or self.head_y < self.wall_size:
            terminal = True
            reward = -1.0

        for x in self.snake_list[:-1]:
            if x == self.snake_head:
                terminal = True
                reward = -1.0

        if not terminal:
            if self.head_x == self.foodx and self.head_y == self.foody:
                self.foodx, self.foody = utils.add_food(self.possible_x, self.possible_y, self.snake_list)
                add_block = True
                self.score += 1
                reward = 10.0  # *(self.score)
            else:
                reward = 0

        if not add_block:
            del self.snake_list[0]

        new_state = self.get_state_map()

        return new_state, reward, terminal

    def render(self):
        if self.firstcall:
            self.firstcall = False
            pygame.init()
            self.env_clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.width, self.length))
            self.score_font = pygame.font.SysFont("comicsansms", 15)
            pygame.display.set_caption('Snake game test')

        self.display_screen()

    def display_screen(self):
        self.screen.fill(eval(self.settings.wall_color.rgb))
        pygame.draw.rect(self.screen, eval(self.settings.bg_color.rgb), [self.wall_size, self.wall_size, self.width - 2*self.wall_size, self.length - 2*self.wall_size])
        value = self.score_font.render("Your Score: " + str(self.score), True, eval(self.settings.sh_color.rgb))
        self.screen.blit(value, [self.wall_size, self.wall_size])
        pygame.draw.rect(self.screen, eval(self.settings.f_color.rgb), [self.foodx, self.foody, self.snake_size, self.snake_size])
        # display_snake(self.screen, MY_YELLOW, self.snake_size, self.snake_list)
        utils.display_snake(self.screen, eval(self.settings.sh_color.rgb), eval(self.settings.sb_color.rgb), self.snake_size, self.snake_list)
        pygame.display.update()
        self.env_clock.tick(10)




