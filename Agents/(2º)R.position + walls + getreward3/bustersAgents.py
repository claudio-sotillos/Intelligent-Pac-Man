# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from learningAgents import ReinforcementAgent

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class QLearningAgent(BustersAgent):

    # alpha - learning rate
    # epsilon - exploration rate
    # gamma - discount factor
    # numTraining - number of training episodes, i.e.no learning after these many episodes
    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()
        # NEW
        initialize = False    # Manual initialization
        if (len(self.initQtable()) != len (self.q_table) ) or (initialize == True) :
            self.q_table = self.initQtable()
        self.epsilon = 0
        self.alpha = 0
        self.gamma= 0.8
        self.countActions = 0

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []
        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)
        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def getQValue(self, state, action):
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
     	legalActions = state.getLegalActions()
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        legalActions = state.getLegalActions()
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        if len(legalActions)==0:
          return None
        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value
        return random.choice(best_actions)

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

####################################################### MODIFIED FUNCTIONS #####################################################################################

    def getAction(self, state):
            # Pick Action
            legalActions = state.getLegalPacmanActions()
            action = None

            if "Stop" in legalActions:
                legalActions.remove("Stop")

            if len(legalActions) == 0:
                return action

            flip = util.flipCoin(self.epsilon)

            if flip:

                return random.choice(legalActions)

            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        stat = self.computePosition(state)
        act = self.actions[action] #Change the string values of the actions by their numeric values by accesign to the self.actions dictionary
        if reward != 0:
            self.q_table[stat][act] = (1 - self.alpha) * self.q_table[stat][act] + self.alpha * (reward + 0)
        else:
            self.q_table[stat][act] = (1 - self.alpha) * self.q_table[stat][act] + self.alpha * (reward + self.gamma * self.computeValueFromQValues(nextState))

    def computePosition(self, gameState):
        # COMPUTEPOSITION FOR THE CLOSESG_WALL ATTRIBUTES
        state = self.closesg_wall(gameState)
        return  6*state[0] + 2*state[1] +state[2]

    # def computePosition(self, gameState):
    #     #COMPUTEPOSITION FOR THE CLOSESG ATTRIBUTES
    #         state = self.closesg(gameState)
    #         return 3 * state[0] + state[1]

####################################################### OUR FUNCTIONS #####################################################################################

    # Used for initializing the qtable (which was initially the one of tutorial 4) with the dimenions that we want.
    def initQtable (self):
        grid = []
        for i in range(0,18):
            row = [0.0,0.0,0.0,0.0]
            grid.append(row)
        return grid

    # PHASE 1: Function that generates tuples (state, action,nexState, reward)
    def mytuple(self, gameState, gameover, printlist):
        tick = self.countActions
        if not gameover:
            state_tick = gameState
            action = self.getAction(gameState)
            state_next_tick = None
            reward = 0
            tup = [state_tick, action, state_next_tick, reward]

            if tick == 0:  # Initial case
                printlist.append(tup)

            # The final case is at the end of this code file. It is saved on the arff when self.gameOver is True.
            else:  # Other cases
                posteriortick = printlist.pop()
                actualstate = gameState
                posteriortick[2] = actualstate
                posteriortick[3] = self.getReward3(posteriortick[0], posteriortick[2])
                printlist.append(tup)

        else:
            posteriortick = printlist.pop()
            actualstate = gameState
            posteriortick[2] = actualstate
            posteriortick[3] = self.getReward3(posteriortick[0], posteriortick[2])

        self.countActions += 1
        if tick != 0:
            return tuple(posteriortick)
        else:
            # return "The tuple of tick 0 is returned in tick 1"
            posteriortick = printlist.pop()
            printlist.append(posteriortick)
            return tuple(posteriortick)


    # Creates two new attributes. Relative position of the Ghost with respect the Pac-Man  in the x and y axis
    def closesg(self,state):
        gostdist = state.data.ghostDistances
        gostdist = [10000 if x == None else x for x in gostdist]    # Change Nones by 10000
        minindex = gostdist.index(min(gostdist))
        coords = state.getGhostPositions()
        nextg = coords[minindex]
        pacmanpos = state.getPacmanPosition()
        movement = (nextg[0] - pacmanpos[0], nextg[1] - pacmanpos[1])

        # Y
        if movement[1] < 0 :
            y = 2
        elif movement[1] > 0 :
            y = 1
        elif movement[1] == 0:
            y = 0

        #X
        if movement[0] < 0 :
            x = 2
        elif movement[0] > 0 :
            x = 1
        elif movement[0] == 0:
            x = 0

        return [x,y]

    # Creates the same two attributes as the before function (closesg) plus an extra attribute which indicates if there is a Wall between the ghost and the Pac-Man
    def closesg_wall(self,state):
        gostdist = state.data.ghostDistances
        gostdist = [10000 if x == None else x for x in gostdist]    # Change Nones by 10000
        minindex = gostdist.index(min(gostdist))
        coords = state.getGhostPositions()
        nextg = coords[minindex]
        pacmanpos = state.getPacmanPosition()
        movement = (nextg[0] - pacmanpos[0], nextg[1] - pacmanpos[1])

        # There is wall --> w = 1
        # There is no wall --> w = 0

        # Y
        if movement[1] < 0 :
            y = 2
            if state.hasWall(pacmanpos[0], pacmanpos[1] - 1) == True:
                wall_y = 1
            else:
                wall_y = 0
        elif movement[1] > 0 :
            y = 1
            if state.hasWall(pacmanpos[0], pacmanpos[1] + 1) == True:
                wall_y = 1
            else:
                wall_y = 0
        elif movement[1] == 0:
            y = 0
            wall_y = 0
        #X
        if movement[0] < 0 :
            x = 2
            if state.hasWall(pacmanpos[0] - 1, pacmanpos[1]) == True:
                wall_x = 1
            else:
                wall_x = 0
        elif movement[0] > 0 :
            x = 1
            if state.hasWall(pacmanpos[0] + 1, pacmanpos[1]) == True:
                wall_x = 1
            else:
                wall_x = 0
        elif movement[0] == 0:
            x = 0
            wall_x = 0

        # XOR The attribute will be 1 if there is a wall in any of the directions Pac_Man is moving
        if (wall_x == 1) or (wall_y == 1):
            w = 1
        else:
            w = 0
        return [x,y,w]

    #  REWARDS
    def getReward(self, state, nextState):
        #  APPROACH FOR FINISH THE GAME AS SOON AS POSSIBLE
        r = nextState.getScore() - state.getScore()
        if r == 199:
                return 1
        else:
            return 0
    def getReward2(self, state, nextState):
        #  APPROACH FOR FINISH THE GAME AS SOON AS POSSIBLE (We give small rewards when we get close to the nearest ghost)
        r = nextState.getScore() - state.getScore()
        # State
        gostdist = state.data.ghostDistances
        gostdist = [10000 if x == None else x for x in gostdist]    # Change Nones by 10000
        minindex = gostdist.index(min(gostdist))
        coords = state.getGhostPositions()
        nextg = coords[minindex]
        pacmanpos = state.getPacmanPosition()
        # Next State
        gostdist = nextState.data.ghostDistances
        gostdist = [10000 if x == None else x for x in gostdist]    # Change Nones by 10000
        minindex = gostdist.index(min(gostdist))
        coords = nextState.getGhostPositions()
        nextg_nst = coords[minindex]
        pacmanpos_nst  = nextState.getPacmanPosition()
        if r == 199 :
            return 1
        elif abs(nextg[0]-pacmanpos[0]) > abs(nextg_nst[0]-pacmanpos_nst[0]) or abs(nextg[1]-pacmanpos[1]) > abs(nextg_nst[1]-pacmanpos_nst[1]):
            return 0.3
        else:
            return 0

    def getReward3(self, state, nextState):
            #  APPROACH FOR FINISH THE GAME AS SOON AS POSSIBLE
            # (Is exactly the same as the reward before, but giving a reward of 0 when there is a wall between the ghost and the Pac-Man)
            r = nextState.getScore() - state.getScore()
            # State
            gostdist = state.data.ghostDistances
            gostdist = [10000 if x == None else x for x in gostdist]  # Change Nones by 10000
            minindex = gostdist.index(min(gostdist))
            coords = state.getGhostPositions()
            nextg = coords[minindex]
            pacmanpos = state.getPacmanPosition()
            # Next State
            gostdist = nextState.data.ghostDistances
            gostdist = [10000 if x == None else x for x in gostdist]  # Change Nones by 10000
            minindex = gostdist.index(min(gostdist))
            coords = nextState.getGhostPositions()
            nextg_nst = coords[minindex]
            pacmanpos_nst = nextState.getPacmanPosition()
            st = self.closesg_wall(nextState)
            if r == 199:
                return 1
            elif abs(nextg[0] - pacmanpos[0]) > abs(nextg_nst[0] - pacmanpos_nst[0]) or abs(nextg[1] - pacmanpos[1]) > abs(nextg_nst[1] - pacmanpos_nst[1]):
                if st[2] == 1:
                    return 0
                else:
                    return 0.3
            else:
                return 0

    # Extra Reward (programmed just in case we finally decided to try a Pac-Dot approach
    def getRewardmaxscore(self, state, nextState):
        #  APPROACH FOR GETTING MAXIMUM SCORE EATING THE PAC-DOTS FIRST
        r = nextState.getScore() - state.getScore()
        if r > 0:
            if r == 99:
                return 1
            else:
                return 0.5
        else:
            return 0














class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print "---------------- TICK ", self.countActions, " --------------------------"
        # Dimensiones del mapa
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print "Width: ", width, " Height: ", height
        # Posicion del Pacman
        print "Pacman position: ", gameState.getPacmanPosition()
        # Acciones legales de pacman en la posicion actual
        print "Legal actions: ", gameState.getLegalPacmanActions()
        # Direccion de pacman
        print "Pacman direction: ", gameState.data.agentStates[0].getDirection()
        # Numero de fantasmas
        print "Number of ghosts: ", gameState.getNumAgents() - 1
        # Fantasmas que estan vivos (el indice 0 del array que se devuelve corresponde a pacman y siempre es false)
        print "Living ghosts: ", gameState.getLivingGhosts()
        # Posicion de los fantasmas
        print "Ghosts positions: ", gameState.getGhostPositions()
        # Direciones de los fantasmas
        print "Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)]
        # Distancia de manhattan a los fantasmas
        print "Ghosts distances: ", gameState.data.ghostDistances
        # Puntos de comida restantes
        print "Pac dots: ", gameState.getNumFood()
        # Distancia de manhattan a la comida mas cercada
        print "Distance nearest pac dots: ", gameState.getDistanceNearestFood()
        # Paredes del mapa
        print "Map:  \n", gameState.getWalls()
        # Puntuacion
        print "Score: ", gameState.getScore()
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        # self.printInfo(gameState)
        # move = Directions.STOP
        # legal = gameState.getLegalActions(0) ##Legal position from the pacman
        # move_random = random.randint(0, 3)
        # if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        # if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        # if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        # if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        # return move
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman

        # Our code
        pacmanpos = gameState.getPacmanPosition()  # Coordenadas del Pac-Man / Pac-Man coordinates
        ghostpos = gameState.getGhostPositions()  # Coordenadas de los fantasmas / Ghosts coordinates
        ghostdistance = gameState.data.ghostDistances  # Distancia del Pac-Man a los fantasmas / Distance from the Pac-Man to the ghosts
        #  FINDING THE NEAREST GHOST
        # We make a deep copy of the distance list, so that we can modify this new list without altering the original list.
        # When a Ghost is eaten, its distance is set to None. What we do is to eliminate the Nones from the deep.copy and just keep the numbers left.
        # Finally, we do min(deep_copydist) for finding the smallest distance of the Ghost that are still alive and that is our new target.
        deep_copydist = []
        temp = []
        condit = True
        for i in ghostdistance:  # Deep copy of the distance list
            deep_copydist.append(i)

        for i in range(len(ghostdistance)):  # Deletes from the deep copy the Nones
            if ghostdistance[i] == None:
                deep_copydist.remove(None)

        nearestg = min(
            deep_copydist)  # Distancia del Pac-Man al fantasma mas cercano/ Distance from the Pac-Man to the nearest ghost
        index = ghostdistance.index(nearestg)  # Index of this distance in the distance list

        # We keep the index because the index in the distance list and in the coordinate list of the ghost coincide.
        # That is index 0 is for the dark blue Ghost, index 1 is for the red ghost, index 2 is for the light blue ghost and index 3 is for the orange Ghost.
        nearestgcoords = ghostpos[index]
        movement = (nearestgcoords[0] - pacmanpos[0], nearestgcoords[1] - pacmanpos[1])
        # Y
        if movement[1] < 0 and Directions.SOUTH in legal:
            move = Directions.SOUTH
        elif movement[1] > 0 and Directions.NORTH in legal:
            move = Directions.NORTH
        # X
        elif movement[0] < 0 and Directions.WEST in legal:
            move = Directions.WEST
        elif movement[0] > 0 and Directions.EAST in legal:
            move = Directions.EAST

        return move
