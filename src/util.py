from collections import namedtuple

players = {0: 'X', 1: 'O'} # X is Player, O is CPU

'''
help command
'''
def help(game):
    print("Supported Commands: ")
    for command in commands:
        print(command, commands[command])

'''
undo command
'''
def undo(game):
    game.undo() #In place

'''
reset command
'''
def reset(game):
    game.reset() #In place

commands = {'h': help, 'u': undo, 'r': reset} # User commands

'''
Prints Available Actions to terminal, no return
'''
def displayActions(availableActions):
    print("'X' can take these remaining boxes: ")
    print([i for i, action in enumerate(availableActions) if action])

'''
Checks if a coord is in bounds, returns Bool
'''
def validCoord(coord, N):
    row, col = coord
    return(0 <= row < N and 0 <= col < N)

'''
Gets coords from user inputted action, returns 'Cood' Named Tuple
'''
def dcdAction(action, N):
    return Coord(int(action/N), action%N)

'''
Reward function for any action, returns int
'''
def reward(action):
    return 0

'''
Output messages after each turn, no return
'''
def output(T, action, randAction=None):
    message = "Player " + str(players[T]) + " Selected Box #" + str(action)
    if(randAction):
        print(message + ", However a random box (Box #" + str(randAction) + ") was instead selected. The Probability of this happening during any move is: " + str(P))
    else:
        print(message)

'''
Get User Input, returns String
'''
def getInput(game):
    availableActions = game.gameState.actions
    displayActions(availableActions) #Output to terminal
    h, r, u = commands['h'], commands['r'], commands['u']
    userInput = int(input("Enter a command or the ID of a remaining box: "))
    while(not validateInput(game, userInput)):
        print("ERROR: Invalid input. Try again: ")
        action = int(input("Enter the ID of a remaining box: "))
    return userInput

'''
Validates User Input, returns Bool
'''
def validateInput(game, input):
    try:
        input = int(input) #Check if valid action
        N = game.N
        return(0 <= input < N*N and game.gameState.actions[input])
    except:
        return(input in commands) #Check if valid command

'''
Performs user requested task, no returnn
'''
def handleInput(game, validatedInput):
    try:
        game.gameState = game.transition(int(validatedInput))
    except:
        commands[validatedInput](game)

def getGameState(N):
    T       = 0
    board   = [[-1] * N for _ in range(N)]
    actions = [1] * (N*N)
    return GameState(T, board, actions)

'''
GameState Representation
'''
GameState = namedtuple('GameState', 'T board actions')

'''
Coord Representation
'''
Coord = namedtuple('Coord', 'Row Col')
