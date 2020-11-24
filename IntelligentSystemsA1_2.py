#!/usr/bin/python
from heap import Heap
import os
import numpy as np
import random
import json as js
import PIL.Image

class WCell:
    """
    WCell(position_vector, value)

    - value -- cell type (travel cost)

    Maze cell class
    """
    COLORS = {} # TODO
    RES = (16,16)
    MAX_NEIGH = 4

    def __init__(self, position, val=0):
        self.position = np.array(position)
        self.value = val
        self.neighbors = [False for i in range(WCell.MAX_NEIGH)]
        self.is_solution = False

    def cost(self):
        return self.value + 1

    def to_image(self):
        "Create numpy array image representation with resolution WCell.RES"
        # Set image to white
        img = np.ones(WCell.RES) * 255

        # set corners to walls (black color)
        img[0,0] = 0
        img[WCell.RES[0]-1,0] = 0
        img[0,WCell.RES[1]-1] = 0
        img[WCell.RES[0]-1,WCell.RES[1]-1] = 0

        # iterate walls
        for i in range(0, self.MAX_NEIGH):
            # if there is wall
            if not self.neighbors[i]:
                # N-S axle
                if WMaze.MOV[i][0] != 0:
                    pixel_wall = range(0,WCell.RES[1])
                    pixel_row = WCell.RES[0]-1 if WMaze.MOV[i][0] > 0 else 0
                    img[pixel_row, pixel_wall] = 0
                # O-E axle
                else:
                    pixel_wall = range(0,WCell.RES[0])
                    pixel_col = WCell.RES[1]-1 if WMaze.MOV[i][1] > 0 else 0
                    img[pixel_wall, pixel_col] = 0
        return img

    def to_dict(self):
        return {'value': self.value, 'neighbors': self.neighbors}

    def __str__(self):
        pos = str(self.position.tolist())[1:-1]
        return f'"({pos})": ' + str(self.to_dict()).lower()

    def __repr__(self):
        return str(self)

class WMaze:
    """
    WMaze(rows, cols)\n
    input: number of rows and columns of the maze\n\n
    Maze class with Wilson's generator
    """

    ID_MOV = ["N", "E", "S", "O"]

    MOV = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    def __init__(self, rows, cols, filedata=None):
        self.rows = rows
        self.cols = cols
        # In the first case, we receive the rows and columns from the user.
        if filedata is None:
            self.wilsonAlgorithmGen()
        # In the second case, we receive the rows, the columns from the .json file passed.
        else:
            self.from_json_file(filedata)

    def succesor_fn(self, state):
        """
        Generate succesors of a given state

        Returns a list of (mov, state, cost)
        """
        cell = WCell(self.matrix[state[0]][state[1]])
        succesors = []
        for i in range(0, cell.MAX_NEIGH):
            if cell.neighbors[i]:
                succesor_state = tuple(np.array(self.MOV[i]) + cell.position)
                succesors.append((self.ID_MOV[i], succesor_state, cell.cost()))
        return succesors

    def to_json(self):
        "Convert maze to a json string"
        row = js.dumps(self.rows)
        column = js.dumps(self.cols)
        mov = js.dumps(self.MOV)
        idm = js.dumps(self.ID_MOV)

        json = "{\n"
        json += f'"rows": {row},\n"cols": {column},\n"max_n": {WCell.MAX_NEIGH},\n"mov": {mov} ,\n"id_mov": {idm},\n'
        json += '"cells": {'

        #maze cells
        matriz = self.matrix
        pos = ()
        number1 =0
        number2 = 0
        aux = ''
        #extracting elements for each cell and dump
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                pos = matriz[i][j].position
                number1=pos[0]
                number2= pos[1]
                aux += "\n\t" + f'"({number1},{number2})": ' + "{" f'"value": {js.dumps(matriz[i][j].value)},"neighbors": {js.dumps(matriz[i][j].neighbors)}'+ "},"

        #delete the last ','
        aux = aux[0:-1]
        aux += "\n\t}\n}"
        json += aux
        return json

    def json_exp(self, filename="maze.json"):
        "Save json represantation of the maze to a file"
        jfile = open(filename, "w")
        #write json file
        jfile.write(self.to_json())
        jfile.close()

    def to_image(self):
        "Convert WMaze to a numpy array describing image"
        get_pix = lambda r, c: (WCell.RES[0] * r, WCell.RES[1] * c)
        img = np.zeros(get_pix(self.rows, self.cols))

        for row in self.matrix:
            for cell in row:
                # mark cell
                pos = get_pix(*cell.position)
                img[pos[0]:pos[0] + WCell.RES[0], pos[1]:pos[1] + WCell.RES[1]] = cell.to_image()
        return img

    def __reset(self):
        "Reset matrix to empty maze state"
        self.matrix = [[WCell([y, x]) for x in range(0, self.cols)] for y in range(0, self.rows)]

    def wilsonAlgorithmGen(self):
        "Generate maze using Wilson's algorithm"
        self.__reset()
        if self.rows < 1 or self.cols < 1:
            return
        # initialize
        fits_boundary = lambda i: i[0] >= 0 and i[0] < self.rows and i[1] >= 0 and i[1] < self.cols
        visited = [[False for x in range(0, self.cols)] for y in range(0, self.rows)]
        free = [(y, x) for x in range(0, self.cols) for y in range(0, self.rows)]
        # set first random cell
        visited[random.randint(0, self.rows - 1)][random.randint(0, self.cols - 1)] = True

        # iterate
        while len(free) > 0:
            walk = []
            row, col = free.pop(random.randint(0, len(free) - 1))

            # get walk path
            stop = False
            while not stop:
                if visited[row][col]:
                    stop = True
                pair = (row, col)
                if pair in walk:
                    # remove loop
                    walk = walk[:walk.index(pair)]
                walk.append(pair)
                row, col = random.choice(
                    list(filter(fits_boundary, np.array(self.MOV) + self.matrix[row][col].position)))

            # follow path and build maze
            row, col = walk[0]
            for i in range(1, len(walk)):
                visited[row][col] = True
                adj = walk[i]
                try:
                    free.remove(adj)
                except ValueError:
                    pass
                mov = np.array(adj) - self.matrix[row][col].position
                side = self.MOV.index(mov.tolist())
                op_side = self.MOV.index((-1 * mov).tolist())
                self.matrix[row][col].neighbors[side] = True
                row, col = adj
                self.matrix[row][col].neighbors[op_side] = True
                
    def from_json_file(self, data):
        """In this method we read the json file in order to retreive the most important information from it.\n
        These are the value and neighbors variable from each cell, so that we can print the maze."""
        
        with open(data, 'r') as f:
            data = js.loads(f.read())

        self.rows = data['rows']
        self.cols = data['cols']
        self.ID_MOV = data['id_mov']
        self.MOV = data['mov']

        tmp = WCell.MAX_NEIGH
        WCell.MAX_NEIGH = data['max_n']
        self.__reset()
        WCell.MAX_NEIGH = tmp

        for i in data['cells']:
            r,c = i[1:-1].split(',')
            self.matrix[int(r)][int(c)].value = data['cells'][i]['value']
            self.matrix[int(r)][int(c)].neighbors = data['cells'][i]['neighbors']

class STNode:
    """
    STNode(depth, cost, state, parent, action, heuristic, value)

    - state -- (row, col)

    SearchTree Node implementation.
    """
    IDC = 0
    def __init__(self, depth, cost, state, parent, action, heuristic, value):
        self.id = STNode.IDC
        STNode.IDC += 1
        self.depth = depth
        self.cost = cost
        self.state = state  #tupla de estado (celda), desde initial
        self.parent = parent
        if self.parent is not None:
            self.id_parent = parent.id
        else:
            self.id_parent = None
        self.action = action
        self.heuristic = heuristic
        self.value = value

    def __str__(self):
        # [<ID>][<COST>,<ID_STATE>,<ID_PARENT>,<ACTION>,<DEPTH>,<HEURISTIC>,<VALUE>]
        return f"[{self.id}][{self.cost}{self.state}{self.id_parent}{self.action}{self.depth}{self.heuristic}{self.value}]"

    def __int__(self):
        return int(self.value)

    def __gt__(self, other):
        # order by [value][row][col][id]
        if type(other) is STNode:
            if self.id == other.id:
                return False
            else:
                return not (self < other)
        else:
            return self.value > other

    def __lt__(self, other):
        # order by [value][row][col][id]
        if type(other) is STNode:
            As = (self.value, self.state[0], self.state[1], self.id)
            Bs = (other.value, other.state[0], other.state[1], other.id)
            for a,b in zip(As, Bs):
                if a == b:
                    continue
                return a < b
            return False
        else:
            return self.value < other
    
    def __eq__(self, other):
        # same state nodes are equal
        if type(other) is STNode:
            return self.state == other.state
        else:
            return self.state == other

    def __repr__(self):
        return str(self)

class Problem:
    """
    Problem(initial_state: tuple, objective_state: tuple, maze: WMaze)

    Load and solve search tree problem.
    """
    CFRONT = Heap
    ALGORITHM = 'BREADTH'
    LIMIT = 1000000

    def __init__(self, init: tuple, obj: tuple, maze: WMaze):
        self.initial = init
        self.objective = obj
        self.maze = maze
        self.frontier = self.CFRONT()

    def solve(self):
        "Build tree and return solution STNode"
        if self.ALGORITHM == 'DEPTH':
            i = 0
            solution = None
            while solution is None:
                solution = self._solve(i)
                i += 1
            return solution
        else:
            return self._solve()

    def _solve(self, limit=None):
        closed = []

        # root element
        h = self.heuristic(self.initial)
        value = self.algorithmValue(0, 0, h)
        root = STNode(0, 0, self.initial, None, None, h, value)
        self.frontier.push(root)

        solution = None
        while len(self.frontier) > 0:
            nodo = self.frontier.pop()

            if nodo in closed:
                continue

            closed.append(nodo.state)

            if self.goal(nodo.state):
                solution = nodo
                break

            for s in self.maze.succesor_fn(nodo.state):
                h = self.heuristic(s[1])
                depth = nodo.depth + 1
                if depth > self.LIMIT:
                    break
                if limit is not None and depth > limit:
                    break
                cost = nodo.cost + s[2]
                value = self.algorithmValue(depth, cost, h)

                successor = STNode(depth, cost, s[1], nodo, s[0], h, value)
                self.frontier.push(successor)

        return solution

    def algorithmValue(self, depth, cost, heuristic):
        if self.ALGORITHM == 'BREADTH':
            return depth
        elif self.ALGORITHM == 'DEPTH':
            return -depth
        elif self.ALGORITHM == 'UNIFORM':
            return cost
        elif self.ALGORITHM == 'GREEDY':
            return heuristic
        elif self.ALGORITHM == "'A":
            return cost + heuristic
        return depth

    def heuristic(self, state):
        "Calculate heuristic"
        # TODO
        # Heuristic((row,column))= |row-target_row| + |column-target_column|
        return abs(1)

    def goal(self, state):
        "Check if current state is the goal state"
        return tuple(state) == self.objective

    @staticmethod
    def from_json(fn='problem.json'):
        with open(fn, 'w') as pfile:
            json = pfile.read()
        data = eval(json)
        # ignore case and load the values
        for k in data:
            if k.lower() == 'initial':
                initial = tuple(data[k])
            elif k.lower().replace('c','') == 'objetive':
                objective = tuple(data[k])
            elif k.lower() == 'maze':
                maze_file = os.path.join(os.path.dirname(fn), data[k])
        return Problem(initial, objective, WMaze(1,1,maze_file))

def main():
    while True:
        print("""Welcome to our maze program, please, choose an option: \n\t
        1. Run the algorithm. \n\t
        2. Read .json file. \n\t
        3. Close program.""")
        option = int(input())
        #The number of rows and columns are intialized to 1 in order to avoid problems

        positive=False
        if option == 1:
            while positive==False:
                print("Introduce value for rows")
                rows = int(input())
                print("Introduce value for columns")
                cols = int(input())
                if rows > 1 and cols > 1:
                    positive = True
                else:
                    print("Maze dimensions must higher than 1, type them again, please.")

            lab = WMaze(rows, cols)
            print(f'Json file has been created in {os.getcwd()}\n')
            lab.json_exp()

            img = PIL.Image.fromarray(lab.to_image())
            img.show()
        elif option == 2:
            lab = WMaze(1, 1, 'sucesores_10X10_maze.json')
            img = PIL.Image.fromarray(lab.to_image())
            img.show()

        elif option == 3:
            print("Exiting program...")
            break
        else:
            print("You pressed a wrong option... \t Press a key to continue.")

if __name__ == '__main__':
    main()
