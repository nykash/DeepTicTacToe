import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TTTNN(nn.Module):
    def __init__(self, lr=1e-6):
        super().__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.n_actions = 9
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class TransitionMemory(object):
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size,)+input_shape, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,)+input_shape, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        self.mem_cntr = 0
        self.input_shape = input_shape

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def reset(self):
        self.state_memory = np.zeros((self.mem_size,) + self.input_shape, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,) + self.input_shape, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def discount_rewards(self, gamma):
      #  print(self.reward_memory[self.mem_cntr])
        r_sum = 0
        for i in reversed(range(self.mem_cntr)):
            done = self.terminal_memory[i]
            if (done):
                r_sum = 0

            if(done and i != self.mem_cntr-1):
                break

           # else:
              #  self.reward_memory[i] = 0#
            r = self.reward_memory[i]
            r_sum = r_sum * gamma + r

            self.reward_memory[i] = r_sum
           # print(i, r, r_sum)
        T.set_printoptions(precision=4, sci_mode=False)
        np.set_printoptions(precision=4, suppress=True)
      #  print(self.reward_memory[i-1:self.mem_cntr+1], "less gooooo")



class Agent(object):
    def __init__(self, input_shape, gamma, epsilon, lr, batch_size, max_mem_size=100000, eps_end=0.01, eps_dec=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.input_shape = input_shape

        self.p1_mem = TransitionMemory(max_mem_size, input_shape)
        self.p2_mem = TransitionMemory(max_mem_size, input_shape)

        self.policy = TTTNN(lr=lr)

        self.action_space = list(range(9))
        self.avg_loss = 0.0
        self.n = 0
        self.avg_losses = []

    def reset_mem(self):
        self.p1_mem.reset()
        self.p2_mem.reset()

    def store_transition_p1(self, state, action, reward, state_, done):
        self.p1_mem.store_transition(state, action, reward, state_, done)

    def store_transition_p2(self, state, action, reward, state_, done):
        self.p2_mem.store_transition(state, action, reward, state_, done)

    def register_win_p1(self, reward):
        self.p2_mem.terminal_memory[self.p2_mem.mem_cntr - 1] = True
        self.p2_mem.reward_memory[self.p2_mem.mem_cntr - 1] = -reward

        self.p1_mem.discount_rewards(self.gamma)
        self.p2_mem.discount_rewards(self.gamma)

    def register_win_p2(self, reward):
        self.p1_mem.terminal_memory[self.p1_mem.mem_cntr - 1] = True
        self.p1_mem.reward_memory[self.p1_mem.mem_cntr - 1] = -reward

        #print(reward, self.p2_mem.reward_memory[self.p2_mem.mem_cntr-1], "rev broda")

        self.p1_mem.discount_rewards(self.gamma)
        self.p2_mem.discount_rewards(self.gamma)

    def choose_action(self, observation, rule=lambda x: True):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation, observation])
            actions = self.policy.forward(state)[0]
          #  print(actions)
            act = actions.argsort()
            k = 8
            while True:
                if (rule(act[k])):
                    break
                k -= 1

            action = act[k]
            print(action, actions, act)
            return action.item()
      #  print("rando bruh")
        actions = []
        for act in self.action_space:
            if (not rule(act)):
                continue
            actions.append(act)
        return np.random.choice(actions)

    def get_learn_params(self, mem):
        max_mem = min(mem.mem_cntr, self.mem_size)
        if(mem.mem_cntr < self.batch_size):
            return None
        batch = np.random.choice(mem.mem_cntr, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(mem.state_memory[batch])
        new_state_batch = T.tensor(mem.new_state_memory[batch])
        reward_batch = T.tensor(mem.reward_memory[batch])
        terminal_batch = T.tensor(mem.terminal_memory[batch])

        action_batch = mem.action_memory[batch]

        policy_eval = self.policy.forward(state_batch)[batch_index, action_batch]
        policy_next = self.policy.forward(new_state_batch)
        policy_next[terminal_batch] = 0.0
        policy_target = reward_batch + self.gamma * T.max(policy_next, dim=1)[0]

        return policy_eval, policy_target

    def learn(self):
        #  print(self.reward_memory, "reward_Mem")
      #  if self.p1_mem.mem_cntr < self.batch_size and self.p2_mem.mem_cntr < self.batch_size:
          #  print("aborting")
           # return

        self.policy.optimizer.zero_grad()


        p1_eval, p1_target = self.get_learn_params(self.p1_mem) if self.p1_mem.mem_cntr > self.batch_size else (None, None)
        p2_eval, p2_target = self.get_learn_params(self.p2_mem) if self.p2_mem.mem_cntr > self.batch_size else (None, None)

        if(p1_eval is None or p2_eval is None):
            policy_eval = p1_eval if p1_eval is not None else p2_eval
        else:
            policy_eval = T.cat([p1_eval, p2_eval])

        if(p1_target is None or p2_target is None):
            policy_target = p1_target if p1_target is not None else p2_target
        else:
            policy_target = T.cat([p1_target, p2_target])

        if(p1_eval is None and p2_eval is None):
            print("aborting")
            return

       # print(policy_eval, policy_target)

      #  policy_eval = p1_eval
      #  policy_target = p1_target

      #  print(p1_eval.shape, p2_eval.shape, policy_eval.shape)

        self.policy.optimizer.zero_grad()

        if(p1_eval is not None):
            loss1 = self.policy.loss(p1_eval, p1_target)
            loss1.backward()

        if(p2_eval is not None):
            loss2 = self.policy.loss(p2_eval, p2_target)
            loss2.backward()
        else:
            loss2 = loss1

        if(p1_eval is None):
            loss1 = loss2


        loss = loss1 + loss2
        self.n += 1
        self.avg_loss = ((self.avg_loss * (self.n-1)) + loss.item())/self.n
        self.avg_losses.append(self.avg_loss)
        print("avg loss is: "+str(self.avg_loss), "reg loss is "+str(loss.item()))

        with open("losses.txt", "a+") as f:
            f.write(str(self.avg_loss)+"\n")

        self.policy.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        print("learned")

    def save(self, path):
        T.save(self.policy, path)

    def load(self, path):
        self.policy = T.load(path)