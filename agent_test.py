from environment import *
from agents import *
from skimage import transform

# def visiualise_buffer(buffer):
# 	for i in range(len(buffer)):
# 		plt.imshow(buffer[i], cmap=plt.get_cmap('gray'))
# 		plt.axis('off')
# 		plt.show()
#
#
# screen_width = 100
# screen_height = 100
# snake_size = 10
#
# env = SnakeEnvironment_2(screen_width, screen_height, snake_size)
# agent = DQNagent(4, env.states_space.shape, 2000, 32)
#
# state = env.observation()
# agent.next_state_buffer.append(state)
#
# action = agent.act(state)
# new_state, reward, terminal = env.step(action)
#
# agent.next_state_buffer.append(new_state)
#
# visiualise_buffer(agent.state_buffer)
# visiualise_buffer(agent.next_state_buffer)

# final_explr_frame = 100000
#
# # on commence à prendre replay en compte
# replay_start_size = 10000
#
# init_explr = 1.0
# final_explr= 0.5
#
# def get_eps(current_step, terminal_eps=0.01, terminal_frame_factor=50):
# 	"""Use annealing schedule similar like: https://openai.com/blog/openai-baselines-dqn/ .
# 	Args:
# 		current_step (int): Number of entire steps agent experienced.
# 		terminal_eps (float): Final exploration rate arrived at terminal_frame_factor * final_explr_frame.
# 		terminal_frame_factor (int): Final exploration frame, which is terminal_frame_factor * final_explr_frame.
# 	Returns:
# 		eps (float): Calculated epsilon for ε-greedy at current_step.
# 	"""
# 	terminal_eps_frame = final_explr_frame * terminal_frame_factor
#
# 	if current_step < replay_start_size:
# 		eps = init_explr
#
# 	elif replay_start_size <= current_step and current_step < final_explr_frame:
# 		eps = (final_explr - init_explr) / (final_explr_frame - replay_start_size) * (
# 					current_step - replay_start_size) + init_explr
#
# 	elif final_explr_frame <= current_step and current_step < terminal_eps_frame:
# 		eps = (terminal_eps - final_explr) / (terminal_eps_frame - final_explr_frame) * (
# 					current_step - final_explr_frame) + final_explr
# 	else:
# 		eps = terminal_eps
# 	return eps
#
# list_plot = []
# for c_e in range(int(1e7)):
# 	ep = get_eps(c_e)
# 	list_plot.append(ep)
#
# plt.plot(list_plot)
# plt.show()
screen_width = 100
screen_height = 100
snake_size = 10
env = SnakeEnvironment_2(screen_width, screen_height, snake_size)
agent = DQNagent(4, env.states_space.shape, 5000, 32)

agent.model_policy = tf.keras.models.load_model('model/model_policy.h5')
agent.update_weights()
agent.current_eps = 0.0

terminal = False
state = env.reset()

agent.state_buffer.append(state)
agent.next_state_buffer.append(state)

while not terminal:
	env.render()
	action = agent.act(state)
	new_state, reward, terminal = env.step(action)

	state = new_state