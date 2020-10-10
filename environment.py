import random
from utils import *
from tensorflow import keras

#########
# COLORS
#########

GREEN = (0, 255, 0)
MY_YELLOW = (208, 208, 63)
BLACK = (0,0,0)

#############################


class SnakeEnvironment:
	def __init__(self, width, length, snake_size):
		self.actions_space = 4
		assert width%snake_size == 0 and length%snake_size == 0, "width and length must be a multiple of snake_size"
		self.width = width
		self.length = length
		self.states_space = np.zeros((int(width/snake_size), int(length/snake_size)))
		self.snake_size = snake_size
		self.score = 0

		self.possible_x = [i for i in range(0, self.width, snake_size)]
		self.possible_y = [i for i in range(0, self.length, snake_size)]

		self.head_x = random.choice(self.possible_x)
		self.head_y = random.choice(self.possible_y)
		self.snake_head = [self.head_x, self.head_y]
		self.snake_list = [self.snake_head]
		self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)

		self.firstcall = True
		self.env_clock = None
		self.screen = None
		self.score_font = None

	def reset(self):
		self.head_x = random.choice(self.possible_x)
		self.head_y = random.choice(self.possible_y)
		self.snake_head = [self.head_x, self.head_y]
		self.snake_list = [self.snake_head]
		self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)

		board_status = np.copy(self.states_space)
		f_pos_x, f_pos_y = int(self.foodx/self.snake_size), int(self.foody/self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3

		head_pos_x, head_pos_y = int(self.snake_head[0]/self.snake_size), int(self.snake_head[1]/self.snake_size)
		board_status[head_pos_y][head_pos_x] = 1

		categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)

		self.score = 0
		return categorical_2d

	def observation(self):
		board_status = np.copy(self.states_space)
		f_pos_x, f_pos_y = int(self.foodx / self.snake_size), int(self.foody / self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3

		head_pos_x, head_pos_y = int(self.snake_head[0] / self.snake_size), int(self.snake_head[1] / self.snake_size)
		board_status[head_pos_y][head_pos_x] = 1

		categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)

		return categorical_2d

	def get_state_map(self):
		board_status = np.copy(self.states_space)
		f_pos_x, f_pos_y = int(self.foodx / self.snake_size), int(self.foody / self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3

		head_pos_x, head_pos_y = int(self.snake_head[0] / self.snake_size), int(self.snake_head[1] / self.snake_size)
		if head_pos_x >= 0 and head_pos_y >= 0 and head_pos_x < self.width/self.snake_size and head_pos_y < self.length/self.snake_size:
			board_status[head_pos_y][head_pos_x] = 1

		for b in self.snake_list[:-1]:
			pos_x, pos_y = int(b[0] / self.snake_size), int(b[1] / self.snake_size)
			board_status[pos_y][pos_x] = 2

		categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)
		return categorical_2d

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

		if self.head_x >= self.width or self.head_x < 0 or self.head_y >= self.length or self.head_y < 0:
			terminal = True
			reward = -1.0

		for x in self.snake_list[:-1]:
			if x == self.snake_head:
				terminal = True
				reward = -2.0

		if not terminal:
			if self.head_x == self.foodx and self.head_y == self.foody:
				self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)
				add_block = True
				self.score += 1
				reward = 10.0
			else:
				reward = 0.0

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
		self.screen.fill(BLACK)
		value = self.score_font.render("Your Score: " + str(self.score), True, MY_YELLOW)
		self.screen.blit(value, [0, 0])
		pygame.draw.rect(self.screen, GREEN, [self.foodx, self.foody, self.snake_size, self.snake_size])
		display_snake(self.screen, MY_YELLOW, self.snake_size, self.snake_list)
		pygame.display.update()
		self.env_clock.tick(10)
