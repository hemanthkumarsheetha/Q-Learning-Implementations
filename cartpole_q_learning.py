import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.reset()

class Agent:
    def __init__(self):
        self.MAX_STATES = 10**4
        self.GAMMA = 0.9
        self.ALPHA = 0.01

    def max_dict(self,d):
        max_v = float('-inf')
        for key , val in d.items():
            if val > max_v:
                max_v=val
                max_key = key
        return  max_key,max_v

    def create_bins(self):
        bins = np.zeros((4,10))
        bins[0] = np.linspace(-4.8,4.8,10)
        bins[1] = np.linspace(-5,5,10)
        bins[2] = np.linspace(-.418,.418,10)
        bins[3] = np.linspace(-5,5,10)

        return  bins

    def assign_bins(self,observation,bins):
        state = np.zeros(4)
        for i in range(4):
            state[i] = np.digitize(observation[i],bins[i])
        return  state

    def get_state_string(self,state):
        return  ''.join(str(int(i)) for i in state)

    def get_all_states_string(self):
        states = []
        for i in range(self.MAX_STATES):
            states.append(str(i).zfill(4))
        return  states
    def initialize_Q(self):
        Q={}
        all_states = self.get_all_states_string()
        for state in all_states:
            Q[state] = {}
            for action in range(env.action_space.n):
                Q[state][action]=0
        return  Q

    def one_game(self,bins,Q,eps=0.5):
        observation = env.reset()
        done = False
        cnt=0
        state = self.get_state_string(self.assign_bins(observation,bins))
        total_reward = 0

        while not done:
            cnt+=1
            if np.random.uniform() < eps:
                act = env.action_space.sample()
            else:
                act = self.max_dict(Q[state])[0]
            observation,reward,done,_ = env.step(act)
            total_reward += reward

            if done and cnt<200:
                reward = -300
            state_new = self.get_state_string(self.assign_bins(observation,bins))
            a1,max_q = self.max_dict(Q[state_new])
            Q[state][act] += self.ALPHA *(reward+self.GAMMA*max_q - Q[state][act])
            state,act = state_new,act

        return  total_reward,cnt
    def many_games(self,bins,N=10000):
        Q = self.initialize_Q()
        length= []
        reward = []
        for n in range(N):
            eps = 1/np.sqrt(n+1)
            episode_reward,episode_length = self.one_game(bins,Q,eps)
            if n % 100 == 0:
                env.render()
                print(n, '%.4f' % eps, episode_reward)
            length.append(episode_length)
            reward.append(episode_reward)

        return length, reward

    def plot_running_avg(self,totalrewards):
        N = len(totalrewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(totalrewards[max(0, t - 100):(t + 1)])
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()
if __name__ == '__main__':
    agent = Agent()
    bins = agent.create_bins()
    episode_lengths,episode_rewards = agent.many_games(bins)
    agent.plot_running_avg(episode_rewards)