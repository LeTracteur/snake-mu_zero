import numpy as np
from utils import *
import random as rd
##################
# GLOBAL param
##################

green = (0, 255, 0)
my_yellow = (208, 208, 63)
black = (0,0,0)

screen_width = 500
screen_height = 500
snake_size = 10
snake_body = 1

score = 0


clock = pygame.time.Clock()
snake_speed = 15

##################
# Game init
##################

pygame.init()

score_font = pygame.font.SysFont("comicsansms", 15)

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Snake game test')
game_over = False

possible_x = [i for i in range(0, screen_width, snake_size)]
possible_y = [i for i in range(0, screen_height, snake_size)]

head_x = rd.choice(possible_x)
# head_x = 500/2
head_y = rd.choice(possible_y)
# head_y = 500/2
snake_head = [head_x, head_y]
snake_list = [snake_head]

foodx, foody = add_food(possible_x, possible_y, snake_list)

x_change = 0
y_change = 0
add_block = False

while not game_over:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			game_over = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				x_change = -snake_size
				y_change = 0
			elif event.key == pygame.K_RIGHT:
				x_change = snake_size
				y_change = 0
			elif event.key == pygame.K_UP:
				y_change = -snake_size
				x_change = 0
			elif event.key == pygame.K_DOWN:
				y_change = snake_size
				x_change = 0

	value = score_font.render("Your Score: " + str(score), True, my_yellow)

	if head_x >= screen_width or head_x < 0 or head_y >= screen_height or head_y < 0:
		game_over = True
		break

	head_x += x_change
	head_y += y_change
	snake_head = [head_x, head_y]
	snake_list.append(snake_head)
	if not add_block:
		del snake_list[0]
	add_block = False

	screen.fill(black)
	screen.blit(value, [0, 0])
	pygame.draw.rect(screen, green, [foodx, foody, snake_size, snake_size])

	for x in snake_list[:-1]:
		if x == snake_head:
			game_over = True

	display_snake(screen, my_yellow, snake_size, snake_list)

	pygame.display.update()

	if head_x == foodx and head_y == foody:
		foodx, foody = add_food(possible_x, possible_y, snake_list)
		add_block = True
		score += 1

	clock.tick(snake_speed)

pygame.quit()
quit()