import random
from utils import *
from tensorflow import keras
from skimage import transform
#########
# COLORS
#########

GREEN = (0, 255, 0)
HEAD_YELLOW = (255, 255, 0)
BODY_YELLOW = (255, 255, 102)
WALL = (255, 0, 0)
BLACK = (0, 0, 0)

GREEN_ID = 3
HEAD_YELLOW_ID = 1
BODY_YELLOW_ID = 2
WALL_ID = 4
BLACK_ID = 0

#############################


class SnakeEnvironment:
	def __init__(self, width, length, snake_size, wall_size):
		self.actions_space = 4
		assert width%snake_size == 0 and length%snake_size == 0, "width and length must be a multiple of snake_size"
		self.wall_size = wall_size
		self.width = width - 2*wall_size
		self.length = length - 2*wall_size
		self.states_space = np.zeros((width, length), dtype=np.uint8)

		self.states_space[:, 0:self.wall_size] = 4
		self.states_space[:, -self.wall_size:] = 4
		self.states_space[-self.wall_size:, :] = 4
		self.states_space[0:self.wall_size, :] = 4

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

		second_x = 2*self.wall_size if self.head_x == self.wall_size else self.head_x - self.wall_size
		second_y = self.head_y
		chunck = [second_x, second_y]
		self.snake_list.insert(0, chunck)

		self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)

		self.board_status = np.copy(self.states_space)

		for i in range(self.snake_size):
			for j in range(self.snake_size):
				self.board_status[self.foody+i][self.foodx+j] = GREEN_ID
				self.board_status[self.snake_head[1]+i][self.snake_head[0]+j] = HEAD_YELLOW_ID
				for b in self.snake_list[:-1]:
					self.board_status[b[1]+i][b[0]+j] = BODY_YELLOW_ID

		self.score = 0

		# self.board_status = scale_lumininance(self.board_status)/255.0
		# self.board_status = transform.resize(self.board_status, (84, 84))
		return self.board_status

	def observation(self):
		self.board_status = np.copy(self.states_space)
		for i in range(self.snake_size):
			for j in range(self.snake_size):
				self.board_status[self.foody+i][self.foodx+j] = GREEN_ID
				self.board_status[self.snake_head[1]+i][self.snake_head[0]+j] = HEAD_YELLOW_ID

		# self.board_status = scale_lumininance(self.board_status)/255.0

		return self.board_status

	def get_state_map(self):
		self.board_status = np.copy(self.states_space)

		for i in range(self.snake_size):
			for j in range(self.snake_size):
				self.board_status[self.foody + i][self.foodx + j] = GREEN_ID
				for b in self.snake_list[:-1]:
					self.board_status[b[1]+i][b[0]+j] = BODY_YELLOW_ID

		if 0 <= self.snake_head[0] < self.width+2*self.wall_size and 0 <= self.snake_head[1] < self.length+2*self.wall_size:
			for i in range(self.snake_size):
				for j in range(self.snake_size):
					self.board_status[self.snake_head[1] + i][self.snake_head[0] + j] = HEAD_YELLOW_ID

		# self.board_status = scale_lumininance(self.board_status)/255.0
		# self.board_status = transform.resize(self.board_status, (84, 84))
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

		if self.head_x >= self.width+self.wall_size or self.head_x < self.wall_size or self.head_y >= self.length + self.wall_size or self.head_y < self.wall_size:
			terminal = True
			reward = -1.0

		for x in self.snake_list[:-1]:
			if x == self.snake_head:
				terminal = True
				reward = -1.0

		if not terminal:
			if self.head_x == self.foodx and self.head_y == self.foody:
				self.foodx, self.foody = add_food(self.possible_x, self.possible_y, self.snake_list)
				add_block = True
				self.score += 1
				reward = 10.0#*(self.score)
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
			self.screen = pygame.display.set_mode((self.width+2*self.wall_size, self.length+2*self.wall_size))
			self.score_font = pygame.font.SysFont("comicsansms", 15)
			pygame.display.set_caption('Snake game test')

		self.display_screen()


	def display_screen(self):
		self.screen.fill(WALL)
		pygame.draw.rect(self.screen, BLACK, [self.wall_size, self.wall_size, self.width, self.length])
		value = self.score_font.render("Your Score: " + str(self.score), True, HEAD_YELLOW)
		self.screen.blit(value, [7, 7])
		pygame.draw.rect(self.screen, GREEN, [self.foodx, self.foody, self.snake_size, self.snake_size])
		# display_snake(self.screen, MY_YELLOW, self.snake_size, self.snake_list)
		display_snake(self.screen, HEAD_YELLOW, BODY_YELLOW, self.snake_size, self.snake_list)
		pygame.display.update()
		self.env_clock.tick(10)
