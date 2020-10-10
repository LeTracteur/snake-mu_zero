import pygame
import numpy as np
from copy import deepcopy


def display_snake(screen, snake_color, snake_vol, snake_list):
	for b in snake_list:
		pygame.draw.rect(screen, snake_color, [b[0], b[1], snake_vol, snake_vol])


def add_food(possible_x, possible_y, snake_list):
	c_possible_x, c_possible_y = deepcopy(possible_x), deepcopy(possible_y)
	for b in snake_list:
		if b[0] in c_possible_x:
			c_possible_x.remove(b[0])
		if b[1] in c_possible_y:
			c_possible_y.remove(b[1])
	pos_x = np.random.choice(np.array(c_possible_x))
	pos_y = np.random.choice(np.array(c_possible_y))
	return pos_x, pos_y


def printProgressBar(iteration, total, eps, tt_reward, score, prefix='', suffix='', decimals=1, length=100, fill='█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	eps = round(eps, 3)
	print('\r%s |%s| %s%% %s | eps %s | total reward %s | score %s' % (prefix, bar, percent, suffix, eps, tt_reward, score), end='\r')
	# Print New Line on Complete
	if iteration == total:
		print()
