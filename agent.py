import tensorflow as tf
import numpy as np

@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class DEEPQ(tf.Module):
    def __init__(self, q_func, observation_shape, num_actions, lr=5e-4, gamma=1.0, double_q=True, grad_norm_clipping=None):
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.optimizer = tf.keras.optimizers.Adam(lr)
        with tf.name_scope('q_network'):
            self.q_network = q_func(observation_shape, num_actions)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_func(observation_shape, num_actions)
        self.eps = tf.Variable(0., name='eps')
        self.double_q = double_q

    @tf.function
    def step(self, obs, stochastic=True, update_eps=-1):
        q_values = self.q_network(obs)
        deterministic_actions = tf.argmax(q_values, axis=1)
        batch_size = tf.shape(obs)[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
        # 以一定的几率将随机的exploration得到的action替换当前Q-value最大的的action
        if stochastic:
          output_actions = stochastic_actions
        else:
          output_actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        return output_actions


    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights):
        with tf.GradientTape() as tape:
            q_t = self.q_network(obs0)
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)
            # q_t_selected 每个sample采取的actions的Q-value
            q_tp1 = self.target_q_network(obs1)

            # double_q 与 非double_q的区别就是用target还是用q_network来确定t+1时刻的action
            if self.double_q:
                q_tp1_using_online_net = self.q_network(obs1)
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(
                    q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)

            dones = tf.cast(dones, q_tp1_best.dtype) # done默认是bool类型 转化成float [0.0, 1.0]
            q_tp1_best_masked = (1.0 - dones) * q_tp1_best  # 如果下一步行为导致游戏结束则直接将下一步的Q-value置0

            # Q(s, a) - (r + gamma * max_a' Q'(s', a')) TD-Error公式
            q_t_selected_target = rewards + self.gamma * q_tp1_best_masked

            # 这里要注意q_t_selected_target应该是作为训练的标签，不参与反向传播，不做stop_gradient处理的话 反向传播会影响梯度的正确推导
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights * errors)
        grads = tape.gradient(weighted_error, self.q_network.trainable_variables)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            clipped_grads = grads
        grads_and_vars = zip(grads, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return td_error

    # 将q_network的参数复制给target_network
    @tf.function(autograph=False)
    def update_target(self):
      q_vars = self.q_network.trainable_variables
      target_q_vars = self.target_q_network.trainable_variables
      for var, var_target in zip(q_vars, target_q_vars):
        var_target.assign(var)
