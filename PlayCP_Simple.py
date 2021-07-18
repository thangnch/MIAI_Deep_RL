import gym
import random
import matplotlib.pyplot as plt
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size):
        self.save_path = "CP.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.discount = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(units = 32, activation="relu", input_dim=self.state_size))
        model.add(Dense(units = 32, activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model


    def save_model(self):
        self.model.save(self.save_path)

    def action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        act_value = np.argmax(act_values[0])
        return act_value

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.discount * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, verbose=0, epochs=1)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class Game():
    def __init__(self):

        self.sample_batch_size = 128
        self.episodes = 500
        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def view(self):
        state = self.env.reset()
        print(state)
        print(self.env.observation_space)
        print(self.env.action_space.n)

    def run(self):
        total_reward = []
        for ep in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            ep_reward = 0
            while not done:
                # self.env.render()
                action = self.agent.action(state)

                next_state, reward, done, _  = self.env.step(action)
                next_state = np.reshape(next_state,[1, self.state_size])

                # Add to hisstory

                self.agent.update(state, action, reward, next_state, done)

                # Chuyển state
                state = next_state
                ep_reward += reward


            print("Episode {}# Score: {}".format(ep, ep_reward))
            total_reward.append(ep_reward)


        # save after all ep
        plot_res(total_reward, '')
        self.agent.save_model()


def plot_res(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(500, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(500, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    game = Game()
    game.run()