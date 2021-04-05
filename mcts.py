import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, applications
import numpy as np
from game import TicTacToe
import math

class Agent(object):
    def __init__(self, model):
        self.dataset = [] # board state, policy (actions), color (-1 or 1), z (whether won or lost game), whether z has been calculated or not
        self.model = model

    def predict(self, single_state):
        return np.array(self.model(single_state[np.newaxis, ...]))[:, 0]

    def store_transition(self, state, pi, color):
        self.dataset.append([state, pi, color, 0, False])

    def update_z(self, z):
        # z is terminal from view of player 1
        for i in reversed(range(len(self.dataset))):
            if(self.dataset[i][4]):
                break
            self.dataset[i][3] = z * self.dataset[i][2]
            self.dataset[i][4] = True

    def save(self, path):
        self.model.save(path)


    def learn(self):

        states = np.array([self.dataset[i][0] for i in range(len(self.dataset))], dtype=np.float32)
        policies = np.array([self.dataset[i][1] for i in range(len(self.dataset))], dtype=np.float32)
        zs = np.array([self.dataset[i][3] for i in range(len(self.dataset))], dtype=np.float32)

       # print(policies)
        self.model.fit(x=states, y=[policies, zs], epochs=10)

        self.dataset = []

class Node(object):
    def __init__(self, state, move=None, color=None):
        self.state = state.clone()
        self.move = move

        if(self.move is not None):
            self.state.move(self.move, color)

        self.n = 1
        self.q = 0
        self.pi = 1

        self.children = []

    def get_puct(self, parent_n):
        return (self.q/self.n) + self.pi * math.sqrt(parent_n)/self.n

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, color):
        for move in self.state.get_legal_moves(color):
            self.children.append(Node(self.state, move, color))

    def set_pi(self, pis):
        for child in self.children:
            child.pi = pis[child.move]


class MCTS(object):
    def __init__(self, agent:Agent, game_state):
        self.root = Node(game_state)
        self.agent = agent

    def set_root(self, game_state):
        self.root = Node(game_state)

    def search(self, color, iters=600):
        for i in range(iters):
            self.iter_through(color)

        visits = [0 for i in range(9)]
        for child in self.root.children:
            visits[child.move] = child.n

        return np.array(visits)/sum(visits)

    def iter_through(self, color):
        passed_nodes = [self.root]
        c = color

        while True:
            if(passed_nodes[-1].is_leaf()):
                break

            if(passed_nodes[-1].state.get_terminal(c)):
                break

            max_puct = float("-inf")
            max_puct_index = 0
            for i, child in enumerate(passed_nodes[-1].children):
                puct = child.get_puct(passed_nodes[-1].n)
                if(puct > max_puct):
                    max_puct = puct
                    max_puct_index = i

            passed_nodes.append(passed_nodes[-1].children[max_puct_index])
            c = -c

        leaf = passed_nodes[-1]
        leaf.expand(c)

        pi, value = self.agent.predict(leaf.state.preprocess(color))
        if(leaf.state.get_terminal(-c) is not None):
            value = -leaf.state.get_terminal(-c)

        leaf.set_pi(pi)

        value = -value

        for node in reversed(passed_nodes):
            node.q += value
            node.n += 1

            value = -value




inp = layers.Input(shape=(9,))
x = layers.BatchNormalization()(inp)
x = layers.Dropout(0.2)(inp)
x = layers.Dense(100)
x = layers.BatchNormalization()(inp)
x = layers.Dropout(0.2)(inp)
x = layers.Dense(50)(x)
pi = layers.Dense(9, activation='softmax', name='pi')(x)   # batch_size x self.action_size
v = layers.Dense(1, activation='tanh', name='v')(x)                    # batch_size x 1

model = models.Model(inputs=inp, outputs=[pi, v])
model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=optimizers.Adam(1e-4))

agent = Agent(model)

episodes = 10000

for i in range(episodes):
    game = TicTacToe()
    mcts = MCTS(agent, game)
    print("episode "+str(i))

    while True:
        mcts.set_root(game)
        p1_probabilities = mcts.search(1, iters=200)
        p1_action = np.random.choice(range(9), p=p1_probabilities)

        agent.store_transition(game.preprocess(1), p1_probabilities, 1)
        game.move(p1_action, 1)

        terminal = game.get_terminal(1)
        if terminal is not None:
            agent.update_z(terminal)
            break


        mcts.set_root(game)
        p2_probabilities = mcts.search(-1, iters=200)
        p2_action = np.random.choice(range(9), p=p2_probabilities)

        agent.store_transition(game.preprocess(-1), p2_probabilities, -1)
        game.move(p2_action, -1)

        terminal = game.get_terminal(-1)
        if terminal is not None:
            agent.update_z(-terminal)
            break


    if(i % 100 == 0):
        agent.learn()
        print("learning")

    agent.save("best_mcts_model.h5")


