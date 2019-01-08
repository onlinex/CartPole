import numpy as np
import tensorflow as tf
import random
import gym

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class neuralNet:
    def __init__(self, lr, s_size, a_size, h_size):
        self.sess = tf.InteractiveSession()

        # feed forward network
        self.state_in = tf.placeholder(tf.float32, [None, s_size])
        self.hidden = tf.contrib.slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = tf.contrib.slim.fully_connected(self.hidden, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output,1)

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        
        # (enumerate integers from 0 to number of iterations in a batch) * num_action + action indexes array
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        # values of taken actions
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

nnet = neuralNet(0.01, 4, 2, 8)

episodes = 5000
iterations = 1000

env = gym.make('CartPole-v0')
gamma = 0.99 # discounted reward function
total_reward = []
total_loss = []

show = False

for ep in range(episodes):
    s = env.reset()
    running_reward = 0
    ep_history = []
    for i in range(iterations):
        if show:
            env.render()

        #Probabilistically pick an action given by network outputs.
        a, av = nnet.sess.run([nnet.chosen_action ,nnet.output], feed_dict={ nnet.state_in: [s] })
        av_chosen = np.random.choice(av[0], p=av[0])
        a = np.argmax(av == av_chosen)

        s1, r, done, _ = env.step(a)
        ep_history.append([s,a,r,s1]) # state, action, reward, new state
        s = s1
        running_reward += r

        if done:
            ep_history = np.array(ep_history)
            ep_history[:,2] = discount_rewards(ep_history[:,2]) # pass all rewards to the function

            feed_dict={
                nnet.state_in: np.vstack(ep_history[:,0]), # states
                nnet.action_holder: ep_history[:,1], # actions
                nnet.reward_holder: ep_history[:,2] # rewards
                }
            for x in range(10):
                nnet.sess.run(nnet.train_step, feed_dict = feed_dict)
                loss_v = nnet.loss.eval(feed_dict = feed_dict)

            total_reward.append(running_reward)
            total_loss.append(loss_v)
            break

    if ep % 50 == 0:
        mtr = np.mean(total_reward[-50:])
        mtl = np.mean(total_loss[-50:])
        if mtr > 190:
            show = True
        print("mean reward: " + str(mtr) + "; mean loss: " + str(mtl))
        
        
