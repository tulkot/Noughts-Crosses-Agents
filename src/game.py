from tictactoe import TicTacToe
from agents import *
from util import *

AGENT = ExpectimaxAgent

def run(game, agent):
    if(game.gameState.T == 1):
        '''
        CPU TURN
        '''
        print("CPU Turn: ")
        action = agent.getMove(game.gameState, 0.95, 0.0, 4)
        game.gameState = game.transition(action)
        return
    '''
    Player Turn
    '''
    print("Your Turn: ")
    cmd = getInput(game)
    handleInput(game, cmd)

if(__name__ == "__main__"):
    N = 0
    while(N < 3):
        try:
            N = int(input("Enter Board Size: "))
        except NameError:
            print("ERROR: Not a number")
            continue
        except SyntaxError:
            print("ERROR: No input")
            continue
        if(N < 3):
            print("ERROR: Can't play with a board size less than 3")
    K = 0
    while(K < 3 or K > N):
        try:
            K = int(input("Enter the number of adjacent squares needed to win: "))
        except NameError:
            print("ERROR: Not a number")
            continue
        except SyntaxError:
            print("ERROR: No input")
            continue
        if(K > N):
            print("ERROR: Can't be greater than board size")
        elif(K < 3):
            print("ERROR: Must be at least 3")
    game = TicTacToe(N, K)
    print(game)
    help(game)
    agent = AGENT(K)
    won, winner = False, None
    while(not(won) and sum(game.gameState.actions) > 0):
        run(game, agent)
        won, winner = agent.check(game.gameState)
        print("CURRENT GAME STATE\n%s"%game)
    if(not won): #Tie
        print("Stalemate.")
    elif(winner == 0): #Player Won
        print("Victory is Yours!!!")
    else:              #CPU Won
        print("FOOLISH MORTAL. TRY HARDER NEXT TIME.")
