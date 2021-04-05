import numpy as np


class TicTacToe(object):
    def __init__(self):
        self.board = np.zeros((3, 3))

    def move(self, action, color):
        y = int(action/3)
        x = action % 3

        self.board[y][x] = color

    def get_terminal(self, view):
        for i in range(3):
            if(sum(self.board[i]) == view*3):
                return 1

            if(sum(self.board.T[i]) == view*3):
                return 1

        if(self.board[0, 0] + self.board[1, 1] + self.board[2, 2] == view*3):
            return 1

        if(self.board[2, 0] + self.board[1, 1] + self.board[0, 2] == view*3):
            return 1

        if(np.all(self.board != 0)):
            return 0

        return None

    def get_legal_moves(self, color):
        legal_moves = np.zeros_like(self.board)
        legal_moves[self.board == 0] = 1

        legal_moves = legal_moves.flatten().astype(bool).tolist()

        res = []
        for i, m in enumerate(legal_moves):
            if(m):
                res.append(i)

        return res

    def clone(self):
        game = TicTacToe()
        game.board = self.board.copy()

        return game

    def preprocess(self, color):

        return self.board.flatten().astype(np.float32) * color

    def print_(self):
        dic = {-1:"o", 1:"x", 0:"h"}
        for i in range(3):
            for j in range(3):
                print(dic[self.board[i][j]], end="")
            print()
