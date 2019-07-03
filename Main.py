from Env import Env
from ActorCritic import *
import tensorflow as tf
import numpy as np
from os.path import isfile
from os import environ
from tflearn import is_training
import pickle

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

actorLR = 0.0001
actorTAU = 0.125
batchSize = 10000

criticLR = 0.001
criticTAU = 0.125
criticGAMMA = 0.5

max_episodes = 1000000
ep_len = 1000
minibatchSize = 10000

random_set = 10000
buff_size = 10000000

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars



def train(sess, actor, critic, noise=None):
	summary_ops, summary_vars = build_summaries()
	sess.run (tf.global_variables_initializer())
	writer = tf.summary.FileWriter('summary', sess.graph)  # summary_dir

	actor.update_target_network()
	critic.update_target_network()
	replay_buffer = ReplayBuffer(buff_size)

	actor.load()
	critic.load()
	if isfile('./buffer.pickle'):
		print('Loading buffer')
		try:
			replay_buffer.load ('./buffer.pickle')
		except:
			print('Buffer pickle error')

	is_training (True)

	for episode_num in range(max_episodes):
		if episode_num % 2 == 0 and episode_num != 0:
			env = Env(player='bot', render=True)
			if episode_num % 500 == 0 and episode_num > random_set:
				actor.save()
				critic.save()
				if episode_num % 1000 == 0:
					replay_buffer.save('./buffer.pickle')
		else:
			env = Env(player='bot', render=False)
		s = env.get_obs
		ep_reward = 0
		ep_ave_max_q = 0
		for episode_iteration in range(ep_len):
			if episode_num > random_set:
				a = actor.predict(np.reshape(s, (1, *env.obs_space)))[0] + actor_noise()
				s2, r, d, info = env.step(np.argmax(a))
				replay_buffer.add(
					np.reshape(s, env.obs_space),
					np.reshape(a, env.action_space),
					r, d,
					np.reshape(s2, env.obs_space),
				)

				if info != 'MOVED' and info != 'WRONG':
					print(info)

				if replay_buffer.size() > minibatchSize:
					s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(minibatchSize)

					target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

					y_i = []
					for k in range(minibatchSize):  # minibatchSize
						if t_batch[k]:
							y_i.append(r_batch[k])
						else:
							y_i.append(r_batch[k] + critic.gamma * target_q[k])

					# Update the critic given the targets
					predicted_q_value, _ = critic.train(
						s_batch, a_batch, np.reshape(y_i, (minibatchSize, 1)))  # (minibatchSize, 1)

					ep_ave_max_q += np.amax(predicted_q_value)

					# Update the actor policy using the sampled gradient
					a_outs = actor.predict(s_batch)
					grads = critic.action_gradients(s_batch, a_outs)
					actor.train (s_batch, grads[0])

					# Update target networks
					actor.update_target_network()
					critic.update_target_network()

				s = s2
				ep_reward += r
				if d:
					summary_str = sess.run (summary_ops, feed_dict={
						summary_vars[0]: ep_reward,
						summary_vars[1]: ep_ave_max_q / float (episode_iteration + 1)
					})

					writer.add_summary(summary_str, episode_num)
					writer.flush()

					print('\033[92m' + '| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} \033[0m'.format(int (ep_reward),
																										episode_num, (
																												ep_ave_max_q / float (
																											episode_iteration + 1))))
					break
			else:
				a = env.action_sample
				s2, r, d, info = env.step(np.argmax(a))
				replay_buffer.add (
					np.reshape(s, env.obs_space),
					np.reshape(a, env.action_space),
					r, d,
					np.reshape(s2, env.obs_space),
				)
				s = s2.copy()
				if d:
					break

with tf.Session() as sess:
	e = Env(render=False)
	actor = Actor(sess, e.obs_space, e.action_space[0], e.action_bound, actorLR, actorTAU, batchSize)
	critic = Critic(sess, e.obs_space, e.action_space[0], criticLR, criticTAU, criticGAMMA, actor.get_num_trainable_vars())
	actor_noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(4))
	train(sess, actor, critic, actor_noise)