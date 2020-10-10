from environment import *
from agents import *

##############################
# Game and Training parameters
##############################

screen_width = 100
screen_height = 100
snake_size = 10

nb_episodes = 5000
steps = 1000

c1 = 10
c2 = 100

###############
# global param
###############

reward_list = []
score_list = []
steps_list = []
tot_reward = 0
###############

env = SnakeEnvironment(screen_width, screen_height, snake_size)
agent = DQNagent(4, env.states_space.shape, 2000, 32)

state = env.observation()

for ep in range(nb_episodes):
	for step in range(steps):
		# env.render()
		action = agent.act(state)
		new_state, reward, terminal = env.step(action)
		reward_list.append(reward)

		tot_reward += reward

		agent.add_to_memory(state, action, reward, new_state, terminal)

		if terminal:
			score_list.append(env.score)
			steps_list.append(step)
			agent.update_epsilon()
			break

		state = new_state

		if ep > agent.batch_size:
			if ep % c1 == 0:
				agent.optimize()

		if ep % c2 == 0:
			agent.update_weights()

		printProgressBar(ep * steps + step, steps * nb_episodes, agent.current_eps, tot_reward, env.score,
		                 prefix="Progress:")

	if ep > agent.batch_size:
		agent.optimize()
	agent.update_weights()

	env.reset()
	state = env.observation()

plt.plot(agent.agent_loss)
plt.title('agent loss')
plt.ylabel('Loss value')
plt.show()

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