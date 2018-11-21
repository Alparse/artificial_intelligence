import gym
import tensorflow as tf

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

def basic_policy(obs):
    angle=obs[2]
    return 0 if angle <0 else 1
rewards=0

n_max_steps = 1000
frames = []

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = basic_policy(observation)
        observation, reward, done, info = env.step(action)
        rewards=rewards+reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.reset()
            print(rewards)
            break
env.close()




