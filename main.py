from agent import *
import gym
from critic import *
from buffer import ReplayBuffer
from scheduler import *
import  numpy as np
import os.path as osp
import logger
import matplotlib.pyplot as plt
lr = 5e-4  # 学习率
total_timesteps = 100000  # 模型总共执行的步数
buffer_size = 50000  # replay buffer的size
exploration_fraction = 0.1  #
exploration_final_eps = 0.02  # 随机选取action的概率最小值
train_freq = 1  # 模型训练开时候，更新模型参数的频率
batch_size = 32
print_freq = 10  # 打印模型训练情况的频率
learning_starts = 1000  # 模型刚开始采集数据的次数
gamma = 1.0  # discount factor
target_network_update_freq = 500  # target模型更新的频率
load_path = None
save_path = 'model'

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DEEPQ(MLP, env.observation_space.shape, env.action_space.n, lr=lr, gamma=gamma)
    replay_buffer = ReplayBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=agent)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    agent.update_target()  # 学习开始前先保证target network和q_network参数一致
    episode_rewards = [0.0]
    saved_mean_reward = None
    # 开始搜集数据
    obs = env.reset()
    obs = np.expand_dims(np.array(obs), axis=0)
    reset = True
    duration = []
    episode_start = 0
    episode_end = 0
    for t in range(total_timesteps):
        env.render()
        update_eps = tf.constant(exploration.value(t))
        action = agent.step(tf.constant(obs), update_eps=update_eps)
        action = action[0].numpy() # tensor转换为numpy用于env输入
        reset = False
        new_obs, rew, done, _ = env.step(action)
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        replay_buffer.add(obs[0], action, rew, new_obs[0], float(done))

        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            episode_end = t
            duration.append(episode_end - episode_start)
            episode_start = t
            obs = env.reset()
            obs = np.expand_dims(np.array(obs),axis=0)
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_indxes = np.ones_like(rewards), None
            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones = tf.constant(actions, dtype=tf.int64), tf.constant(rewards), tf.constant(dones)
            weights = tf.constant(weights)

            td_errors = agent.train(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            agent.update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        # todo 每一个episode记录一次
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

    plt.figure()
    plt.plot(len(duration), duration)
    plt.figure()
    plt.plot(len(episode_rewards), episode_rewards)
    plt.show()