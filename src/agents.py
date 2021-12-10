import numpy as np
import copy
from util import *

vals = {}
qVals = {}

'''
Abstract Computer Agent
'''
class Agent():
    def __init__(self, K):
        self.K = K

    def checkBox(self, board): #Checks one sub-grid
        N = len(board)
        '''
        Horizontal Check
        '''
        for row in range(N):
            if(board[row][0] >= 0 and all(board[row][col] == board[row][col+1] for col in range(N-1))):
                return True, board[row][0]
        '''
        Vertical Check
        '''
        for col in range(N):
            if(board[0][col] >= 0 and all([board[row-1][col] == board[row][col] for row in range(1, N)])):
                return True, board[0][col]
        '''
        Diagonal 1 Check
        '''
        if board[0][0] >= 0 and all([board[row-1][row-1] == board[row][row] for row in range(1, N)]):
            return True, board[0][0]
        '''
        Diagonal 2 Check
        '''
        if board[0][-1] >= 0 and all([board[i-1][N-i] == board[i][N-i-1] for i in range(1, N)]):
            return True, board[0][-1]
        '''
        Default
        '''
        return False, None

    def check(self, gameState): #Cheks all KXK sub-grids of a gameState
        board = np.matrix(gameState.board)
        N = len(board)
        K = self.K
        for row in range(N-K+1):
            for col in range(N-K+1):
                won, winner = self.checkBox(board[row:row+K,col:col+K].tolist())
                if(won):
                    return True, winner
        '''
        Default
        '''
        return False, None

    def qScore(self, gameState, action, L, P, D, W):
        qState = str(gameState) + str(action)
        if qState in qVals:
            return qVals[qState]
        s = reward(action)
        for state, prob in self.transition(gameState, action, P):
            s += L * prob * self.score(state, L, P, D, W, 0)
        qVals[qState] = s
        return s

    def score(self, gameState, L, P, D, W, d):
        '''
        If the score for a gameState has already been computed, fetch
        '''
        if(str(gameState) in vals):
            return vals[str(gameState)]

        '''
        Check if the game has been won, if so return appropriate value
        '''
        won, winner = self.check(gameState)
        if(won):
            vals[str(gameState)] = W if(winner==0) else -W #Win Score
            return vals[str(gameState)]

        '''
        Check if the depth search limit has been reached, or if there
        are no valid actions left. If so, return 0
        '''
        actions = gameState.actions
        if(sum(actions)==0 or d==D):
            vals[str(gameState)] = 0
            return 0
        return(self.computeScore(gameState, L, P, D, W, d))

    '''
    Agent gets its optimal move
    '''
    def getMove(self, gameState, L=1, P=0, D=5, W=1000):
        minQ, optimalAction = float('inf'), None
        ties = []
        actions = gameState.actions
        for i, action in enumerate(actions):
            if(action):
                q = self.qScore(gameState, i, L, P, D, W)
                if(q < minQ):
                    minQ, optimalAction = q, i
                    ties = []
                elif(q == minQ):
                    ties.append(i)
        if(ties):
            return np.random.choice(ties)
        return optimalAction

    '''
    Returns next game state given a current state and an action
    '''
    def update(self, gameState, action):
        newT = (gameState.T+1)%2
        newBoard = copy.deepcopy(gameState.board)
        N = len(newBoard)
        row, col = dcdAction(action, N)
        newBoard[row][col] = gameState.T
        newActions = list(gameState.actions)
        newActions[action] = 0
        newState = GameState(newT, newBoard, newActions)
        return newState

    '''
    Probability of all resulting states from an action
    '''
    def transition(self, gameState, action, P):
        newState = self.update(gameState, action)
        yield(newState, float(1-P))
        if(P > 0): #P is the chance of a random move being forced
            for randAction, possible in enumerate(actions):
                if(possible):
                    newState = self.update(gameState, randAction)
                    yield(newState, (float(P)/sum(actions))) #Random game state probabilities_

    '''
    Prints values of next possible states for the CPU
    '''
    def printVals(self, gameState):
        V = {}
        for i, action in enumerate(gameState.actions):
            if(action):
                for newState, _ in self.transition(gameState, i):
                    V[str(newState)] = L * self.score(newState)
        print(V)
        return V

'''
Expectimax Computer Agent
'''
class ExpectimaxAgent(Agent):
    def __init__(self, K):
        Agent.__init__(self, K)

    def computeScore(self, gameState, L, P, D, W, d=0): 
        '''
        Last resort: Compute the score of the Game State
        '''
        bestScore, optim = W, min
        if(gameState.T == 0):
            bestScore, optim = -bestScore, max
        accum = 0
        actions = gameState.actions
        for i, action in enumerate(actions):
            if action:
                newState = self.update(gameState, i)
                newScore = reward(action) + L*self.score(newState, L, P, D, W, d+1)
                accum += newScore
                bestScore = optim(bestScore, newScore)
        totalScore = ((1.0-P)*bestScore + (P)*(accum/sum(actions)))
        vals[str(gameState)] = totalScore
        return totalScore

class MinimaxAgent(Agent):
    def __init__(self, K):
        Agent.__init__(self, K)

    def computeScore(self, gameState, L, P, D, W, d=0):
        '''
        Last Resort: Compute the score of the Game State
        '''
        bestScore, optim = W, min
        if(gameState.T == 0):
            bestScore, optim = -bestScore, max
        actions = gameState.actions
        for i, action in enumerate(actions):
            if action:
                newState = self.update(gameState, i)
                newScore = reward(action) + L*self.score(newState, L, P, D, W, d+1)
                bestScore = optim(bestScore, newScore)
        vals[str(gameState)] = bestScore
        return bestScore

class MinimaxAlphaBeta(Agent):
    def __init__(self, K):
        Agent.__init__(self, K)

    def computeScore(self, gameState, L, P, D, W, d, alpha=float('-inf'), beta=float('inf')):
        '''
        Last Resort: Compute the score of the Game State
        '''
        actions = gameState.actions
        if(d==D or sum(actions)==0): return 0
        if(gameState.T == 0): #Maximizing Agent
            bestScore = -W
            for i, action in enumerate(actions):
                if(action):
                    newState = self.update(gameState, i)
                    newScore = reward(action) + L+self.computeScore(newState, L, P, D, W, d+1, alpha, beta)
                    bestScore = max(bestScore, newScore)
                    alpha = max(alpha, newScore)
                    if(alpha > beta): break
        else: #Minimizing Agent
            bestScore = W
            for i, action in enumerate(actions):
                if(action):
                    newState = self.update(gameState, i)
                    newScore = reward(action) + L*self.computeScore(newState, L, P, D, W, d+1, alpha, beta)
                    bestScore = min(bestScore, newScore)
                    beta = min(newScore, beta)
                    if(alpha > beta): break
        vals[str(gameState)] = bestScore
        return bestScore
