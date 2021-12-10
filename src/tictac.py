import numpy as np
from enum import Enum
import copy
from util import *
from agents import Agent

MODE = "three"

N     = 3                 # N X N grid
P     = 0.0               # P(Random)
W     = 1000              # Win Score Reward
L     = 0.95              # Decay
D     = 3                 # Max Depth of Search
T     = 0                 # 0 is Player, 1 is CPU
K     = 3                 # Num square in a row needed to win
qVals = {}                # Cache qVals
vals  = {}                # Vals of states

'''
Game Functions
'''
class TicTacToe():
    def __init__(self, N=N, K=K):
        self.gameState = getGameState(N)
        self.states = [self.gameState]
        self.N = N
        self.K = K

    '''
    Takes in an action, updates gameState accordingly
    '''
    def transition(self, action):
        T, board, actions = self.gameState
        newT, newBoard, newActions = (T+1)%2, copy.deepcopy(board), list(actions)
        randAction = None
        if(np.random.random() < P): #Random action selected
            randAction = np.random.choice([i for i in range(len(actions)) if actions[i]])
        output(T, action, randAction) #Display action with player
        if(randAction):
            action = randAction
        row, col = dcdAction(action, self.N)
        newBoard[row][col] = T
        newActions[action] = 0
        newState = GameState(newT, newBoard, newActions)
        self.states.append(newState)
        return newState

    def undo(self):
        if(len(self.states) < 2):
            raise(Exception("ERROR: Undo attempted but you have not taken any actions yet"))
        self.states.pop()
        self.states.pop()
        self.gameState = self.states[-1]

    def reset(self):
        self.gameState = getGameState(self.N)
        self.states = [self.gameState]

    def __str__(self): #Overwrites toString()
        N = self.N
        cellWidth = 1
        for row in range(N):
            for col in range(N):
                cellWidth = max(cellWidth, len(str(row*N + col)))
        s = ''
        for row in range(N):
            for col in range(N):
                try:
                    numDigits = len(str(self.gameState.board[row][col]))
                    numSpaces, remainder = (cellWidth-numDigits)//2, (cellWidth-numDigits)%2
                    player = str(players[self.gameState.board[row][col]])
                    s += ('[' + numSpaces*' ' + player + numSpaces*' ' + remainder*' ' + ']')
                except:
                    cellName = row*N + col
                    numDigits = len(str(cellName))
                    numSpaces, remainder = (cellWidth-numDigits)//2, (cellWidth-numDigits)%2
                    s += ('[' + numSpaces*' ' + str(cellName) + numSpaces*' ' + remainder*' ' + ']')
            s += ('\n')
        return s
