from game import *
from nn import *
import math


def main(episodes=1):
    T.autograd.set_detect_anomaly(True)
    model = Agent(input_shape=(9,), gamma=0.99, epsilon=0.5, lr=1e-6, batch_size=32, eps_dec=1/10000, eps_end=0.5)
    for i in range(episodes):
        done = False
        game = TicTacToe()
        if(i % 10 == 0):
            model.reset_mem()
        print("new episode"+str(i)+"\n", model.epsilon)
        while (not done):
            state = game.preprocess(1)
            action = model.choose_action(state, rule=lambda x:x in game.get_legal_moves(1))

            game.move(action, 1)

            state_ = game.preprocess(1)
            reward = game.get_terminal(1)
            done = reward is not None
            reward = reward if reward is not None else 0
            model.store_transition_p1(state, action, reward if reward is not None else 0, state_, done)

            if(done):
                game.print_()
                model.register_win_p1(reward)
                continue

            state = game.preprocess(-1)
            action = model.choose_action(state, rule=lambda x:x in game.get_legal_moves(-1))

            game.move(action, -1)

            state_ = game.preprocess(-1)
            reward = game.get_terminal(-1)
            done = reward is not None
            reward = reward if reward is not None else 0
            model.store_transition_p2(state, action, reward, state_, done)

            if (done):
                game.print_()
                model.register_win_p2(reward)
                continue


        model.learn()


    model.save("model.h5")

def test():
    model = Agent(input_shape=(9,), gamma=0.99, epsilon=0.0, lr=1e-8, batch_size=64)

    model.load("model.h5")

    done = False
    game = TicTacToe()

    while (not done):
        state = game.preprocess(1)
        action = model.choose_action(state, rule=lambda x:x in game.get_legal_moves(1))
        game.move(action, 1)
        reward = game.get_terminal(1)

        if(reward is not None):
            if(reward == 0):
                print("TIE")
                done = True
                continue

            if(reward == 1):
                game.print_()
                print("AI WINS")
                done = True
                continue

        game.print_()

        move = input("get ya move bruv ")
        while (not move.isdigit() or (move.isdigit() and int(move) not in game.get_legal_moves(-1))):
            move = input("pick a move that actually does something -_- ")

        move = int(move)
        game.move(move, -1)

        reward = game.get_terminal(-1)

        if (reward is not None):
            if (reward == 0):
                print("TIE")
                done = True
                continue

            if (reward == 1):
                print("HUMAN WINS")
                done = True
                continue
        print(game.preprocess(1), game.preprocess(-1))

def self_play():
    model = Agent(input_shape=(9,), gamma=0.99, epsilon=0.0, lr=1e-8, batch_size=64)

    done = False
    game = TicTacToe()
    while (not done):
        state = game.preprocess(1)
        action = model.choose_action(state, rule=lambda x: x in game.get_legal_moves(1))

        game.move(action, 1)
        game.print_()
        print("\n")
        reward = game.get_terminal(1)

        if (reward is not None):
            if (reward == 0):
                print("TIE")
                done = True
                continue

            if (reward == 1):
                game.print_()
                print("P1 WINS")
                done = True
                continue

        state = game.preprocess(-1)
        action = model.choose_action(state, rule=lambda x: x in game.get_legal_moves(-1))

        game.move(action, -1)
        game.print_()

        reward = game.get_terminal(-1)

        if (reward is not None):
            if (reward == 0):
                print("TIE")
                done = True
                continue

            if (reward == 1):
                print("P2 WINS WINS")
                done = True
                continue






if __name__ == "__main__":
    main(episodes=10000)
    #self_play()
    test()