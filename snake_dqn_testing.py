# from old.environment import *
from environment import *
from agents import *

##############################
# Game and Training parameters
##############################

screen_width = 100
screen_height = 100
snake_size = 10

nb_episodes = 10000
steps = 2000

counter = 0

c2 = 500

###############
# global param
###############

reward_list = []
score_list = []
steps_list = []
t_r = 0
tot_reward = []

best_score = 0
eps_val = 0.0
###############

env = SnakeEnvironment_2(screen_width, screen_height, snake_size)
agent = DQNagent(4, env.states_space.shape, 5000, 32)

agent.model_policy.summary()

# state = env.observation()
# agent.next_state_buffer.append(state)
loss = 0
for ep in range(nb_episodes):
	state = env.reset()
	for _ in range(agent.tau):
		agent.state_buffer.append(state)
		agent.next_state_buffer.append(state)

	for step in range(steps):
		agent.update_epsilon()

		# env.render()
		action = agent.act(state)
		new_state, reward, terminal = env.step(action)

		agent.next_state_buffer.append(new_state)

		reward_list.append(reward)

		t_r += reward
		tot_reward.append(t_r)
		# copy_s_b = np.stack(agent.state_buffer, axis=2)
		# copy_s_n_b = np.stack(agent.next_state_buffer, axis=2)
		agent.add_to_memory(deepcopy(agent.state_buffer), action, reward, deepcopy(agent.next_state_buffer), terminal)

		if terminal:
			score_list.append(env.score)
			steps_list.append(step)
			break

		state = new_state

		if step > agent.batch_size:
			# if ep % c1 == 0:
			loss = agent.optimize()

		if counter % c2 == 0:
			agent.update_weights()

		printProgressBar(ep * steps + step, steps * nb_episodes, agent.current_eps, t_r, env.score, loss,
		                 prefix="Progress:")

		counter += 1

	if ep > agent.batch_size:
		agent.optimize()
	agent.update_weights()

	if env.score > best_score:
		best_score = env.score
		agent.save()
		eps_val = agent.current_eps


agent.plot_loss()

plt.plot(reward_list)
plt.title('reward evolution')
plt.ylabel('tot reward')
plt.show()

plt.plot(steps_list)
plt.title('steps per ep')
plt.ylabel('steps')
plt.xlabel('ep')
plt.show()

plt.plot(score_list)
plt.title('score evolution')
plt.ylabel('final score')
plt.xlabel('game')
plt.show()

agent.save()