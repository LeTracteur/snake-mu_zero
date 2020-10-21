import pygame
import numpy as np
import random
from copy import deepcopy


def display_snake(screen, head_color, body_color, snake_vol, snake_list):
	pygame.draw.rect(screen, head_color, [snake_list[-1][0], snake_list[-1][1], snake_vol, snake_vol])
	for b in snake_list[:-1]:
		pygame.draw.rect(screen, body_color, [b[0], b[1], snake_vol, snake_vol])


def add_food(possible_x, possible_y, snake_list):
	c_possible_x, c_possible_y = deepcopy(possible_x), deepcopy(possible_y)
	all_compo = []
	for x in c_possible_x:
		for y in c_possible_y:
			all_compo.append((x,y))
	for b in snake_list:
		tuple_c = (b[0], b[1])
		if tuple_c in all_compo:
			all_compo.remove(tuple_c)
	pos_x, pos_y = random.choice(all_compo)
	return pos_x, pos_y


def printProgressBar(iteration, total, eps, tt_reward, score, loss, ep, prefix='', suffix='', decimals=1, length=100, fill='█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	eps = round(eps, 3)
	print('\r%s |%s| %s%% %s | loss %s | eps %s | total reward %s | ep %s | score %s' % (prefix, bar, percent, suffix, loss, eps, tt_reward, ep, score), end='\r')
	# Print New Line on Complete
	if iteration == total:
		print()


def scale_lumininance(img):
	return np.dot(img[...,:3], [0.299, 0.587, 0.114])
