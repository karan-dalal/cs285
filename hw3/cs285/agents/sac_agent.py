from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure import sac_utils

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        ac = self.actor(next_ob_no) # Select next action
        ac_select = ac.sample() # Sample next action (from action space)
        log_prob = ac.log_prob(ac_select) # Compute log prob of next action
        log_prob = log_prob.sum(dim=1) # Sum over action dimensions
        q = self.critic_target(next_ob_no, ac_select) # Compute Q value of next action + state
        
        target = re_n  + self.gamma * (1.0 - terminal_n) * (q - self.actor.alpha * log_prob) # Calculate target
        target = target.detach()
        
        critic_loss = self.critic.update(ob_no, ac_na, target)
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging
        critic_loss = 0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        critic_loss /= self.agent_params['num_critic_updates_per_agent_update']
            
        if self.training_step % self.critic_target_update_frequency == 0:
            first_network = self.critic.Q1
            first_tar_network = self.critic_target.Q1
            sac_utils.soft_update(first_network, first_tar_network, self.critic_tau)
            second_network = self.critic.Q2
            second_tar_network = self.critic_target.Q2
            sac_utils.soft_update(second_network, second_tar_network, self.critic_tau)

        actor_loss = 0
        alpha_loss = 0
        self.actor.alpha = 0
        if self.training_step % self.actor_update_frequency == 0:
            alpha = 0
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss_next, alpha_loss_next, alpha_next = self.actor.update(ob_no, self.critic)
                actor_loss += actor_loss_next
                alpha_loss += alpha_loss_next
                alpha += alpha_next
                actor_loss += self.update_actor(ob_no)
            actor_loss /= self.agent_params['num_actor_updates_per_agent_update']
            alpha_loss /= self.agent_params['num_actor_updates_per_agent_update']
            alpha /= self.agent_params['num_actor_updates_per_agent_update']

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
