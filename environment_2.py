import random
from utils import *
from tensorflow import keras

#########
# COLORS
#########

GREEN = (0, 255, 0)
HEAD_YELLOW = (255, 255, 0)
BODY_YELLOW = (255, 255, 102)

WALL = (255, 0, 0)

BLACK = (0, 0, 0)

#############################


class SnakeEnvironment_2:
	def __init__(self, width, length, snake_size, wall_size):
		self.actions_space = 4
		assert width%snake_size == 0 and length%snake_size == 0, "width and length must be a multiple of snake_size"
		self.wall_size = wall_size
		self.width = width - 2*wall_size
		self.length = length - 2*wall_size
		self.states_space = np.zeros((int(width/snake_size), int(length/snake_size)))

		self.states_space[:, 0] = 4.0
		self.states_space[:, -1] = 4.0
		self.states_space[-1, :] = 4.0
		self.states_space[0, :] = 4.0

		self.snake_size = snake_size
		self.score = 0

		self.possible_x = [i for i in range(wall_size, self.width, snake_size)]
		self.possible_y = [i for i in range(wall_size, self.length, snake_size)]

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

		second_x = 2 * self.wall_size if self.head_x == self.wall_size else self.head_x - self.wall_size
		second_y = self.head_y
		chunck = [second_x, second_y]
		self.snake_list.insert(0, chunck)

		self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)

		board_status = np.copy(self.states_space)

		f_pos_x, f_pos_y = int(self.foodx/self.snake_size), int(self.foody/self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3.0

		head_pos_x, head_pos_y = int(self.snake_head[0]/self.snake_size), int(self.snake_head[1]/self.snake_size)
		board_status[head_pos_y][head_pos_x] = 1.0

		b_pos_x = int(self.snake_list[0][0]/self.snake_size)
		b_pos_y = int(self.snake_list[0][1]/self.snake_size)
		board_status[b_pos_y][b_pos_x] = 2.0

		# categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)

		self.score = 0
		# return categorical_2d
		# return np.expand_dims(board_status, axis=2)
		return board_status

	def observation(self):
		board_status = np.copy(self.states_space)
		f_pos_x, f_pos_y = int(self.foodx / self.snake_size), int(self.foody / self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3.0

		head_pos_x, head_pos_y = int(self.snake_head[0] / self.snake_size), int(self.snake_head[1] / self.snake_size)
		board_status[head_pos_y][head_pos_x] = 1.0

		# categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)
		# return categorical_2d
		# return np.expand_dims(board_status, axis=2)
		return board_status

	def get_state_map(self):
		board_status = np.copy(self.states_space)
		f_pos_x, f_pos_y = int(self.foodx / self.snake_size), int(self.foody / self.snake_size)
		board_status[f_pos_y][f_pos_x] = 3.0

		head_pos_x, head_pos_y = int(self.snake_head[0] / self.snake_size), int(self.snake_head[1] / self.snake_size)
		if 0 <= head_pos_x < self.width / self.snake_size and 0 <= head_pos_y < self.length / self.snake_size:
			board_status[head_pos_y][head_pos_x] = 1.0

		for b in self.snake_list[:-1]:
			pos_x, pos_y = int(b[0] / self.snake_size), int(b[1] / self.snake_size)
			board_status[pos_y][pos_x] = 2.0

		# categorical_2d = keras.utils.to_categorical(np.array(board_status), num_classes=4)
		# return categorical_2d
		# return np.expand_dims(board_status, axis=2)
		return board_status

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

		if self.head_x >= self.width+self.wall_size or self.head_x < self.wall_size or self.head_y >= self.length + self.wall_size or self.head_y < self.wall_size:
			terminal = True
			reward = -1

		for x in self.snake_list[:-1]:
			if x == self.snake_head:
				terminal = True
				reward = -1

		if not terminal:
			if self.head_x == self.foodx and self.head_y == self.foody:
				self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)
				add_block = True
				self.score += 1
				reward = 1.0 * (self.score+2)
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
			self.screen = pygame.display.set_mode((self.width+2*self.wall_size, self.length+2*self.wall_size))
			self.score_font = pygame.font.SysFont("comicsansms", 15)
			pygame.display.set_caption('Snake game test')

		self.display_screen()

	def display_screen(self):
		self.screen.fill(WALL)
		pygame.draw.rect(self.screen, BLACK, [self.wall_size, self.wall_size, self.width, self.length])
		value = self.score_font.render("Your Score: " + str(self.score), True, HEAD_YELLOW)
		self.screen.blit(value, [10, 10])
		pygame.draw.rect(self.screen, GREEN, [self.foodx, self.foody, self.snake_size, self.snake_size])
		# display_snake(self.screen, MY_YELLOW, self.snake_size, self.snake_list)
		display_snake(self.screen, HEAD_YELLOW, BODY_YELLOW, self.snake_size, self.snake_list)
		pygame.display.update()
		self.env_clock.tick(10)
