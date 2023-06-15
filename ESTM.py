import numpy as np
import random
import math
from typing import Literal
import colorama
from colorama import Fore, Back, Style

colorama.init()

# Types
Tile = str
Coordinates = tuple[int, int]
Up = tuple[Literal[-1], Literal[0]]
Down = tuple[Literal[1], Literal[0]]
Left = tuple[Literal[0], Literal[-1]]
Right = tuple[Literal[0], Literal[1]]
Direction = Up | Down | Left | Right
Compatibility = tuple[Tile, Tile, Direction]
Weights = dict[Tile, int]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]


class Cell:

    def __init__(self, pos, options: set[Tile]):
        self.position = pos  # position is given as a (y, x) tuple
        self.options = options.copy()  # list of possible options available
        self.collapsed = False  # collapsed state of the cell
        self.value = None  # chosen option for this cell

    def __repr__(self):
        return f'tile {self.position} \n' \
               f'options : {[i for i in self.options]} \n' \
               f'value : {self.value} \n' \
               f'collapsed : {self.collapsed} \n'

    def collapse(self, choice):
        self.value = choice
        self.collapsed = True
        self.options = set(self.value)

    def constrain(self, forbidden: Tile):
        self.options.remove(forbidden)


class WaveFunction:

    def __init__(self, size: tuple[int, int], options, weights, comp: set[Compatibility]):
        self.grid = np.empty(shape=size, dtype=Cell)
        self.size = size
        self.weights: Weights = weights
        self.compatibilities = comp
        for _y in range(size[0]):
            for _x in range(size[1]):
                self.grid[_y, _x] = Cell((_y, _x), options)

    def __repr__(self):
        display = f""
        for _y in range(self.size[0]):
            for _x in range(self.size[1]):
                display += f" {str(self.grid[_y, _x].value):>4} "
            display += "\n"

        return display

    def get(self, coords: Coordinates):
        return self.grid[coords].options

    def check(self, tile1, tile2, direction):
        return (tile1, tile2, direction) in self.compatibilities

    def getEntropyAtCoords(self, coords: Coordinates):
        weights_sum = sum([self.weights[w] for w in self.get(coords)])
        weights_log_sum = sum([self.weights[w] +
                               math.log(self.weights[w])
                               for w in self.get(coords)])
        return math.log(weights_sum) - (weights_log_sum / weights_sum)

    def getMinEntropyCoords(self):
        min_entropy_coords: Coordinates = (0, 0)
        min_entropy = None
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if self.grid[y, x].collapsed:
                    continue
                # TODO : might need to check for options errors while we are looping through the whole array
                entropy = self.getEntropyAtCoords((y, x))
                # Add some noise
                entropy += (random.random() / 1000)
                if min_entropy is None or entropy < min_entropy:
                    min_entropy = entropy
                    min_entropy_coords = (y, x)

        return min_entropy_coords

    def collapseCellAtCoords(self, coords: Coordinates):
        cell: Cell = self.grid[coords]
        weighted_seq = [weight for (item, weight) in self.weights.items() if item in cell.options]
        choice = random.choices(list(cell.options), weighted_seq)
        cell.collapse(choice[0])

    def propagate(self, coords: Coordinates):
        stack = [coords]
        while len(stack) > 0:
            cur_coords = stack.pop()  # get the current coords to analyze
            cur_possible_tiles = self.get(cur_coords)  # get the available tiles
            # TODO: check for possible error ? if no options -> error
            # iterate around current coords
            for d in valid_dirs(cur_coords, self.size):

                other_coords = (cur_coords[0] + d[0], cur_coords[1] + d[1])
                # loop through every tile in other location
                for other_tile in set(self.get(other_coords)):

                    # check if the other tile is compatible with any of the available tiles
                    other_tile_possible = any([
                        self.check(cur_tile, other_tile, d)
                        for cur_tile in cur_possible_tiles
                    ])
                    # If the tile is not compatible with any of the tiles in
                    # the current location then it is impossible
                    # for it to ever get chosen. We therefore remove it from
                    # the other location.
                    if not other_tile_possible:
                        self.grid[other_coords].constrain(other_tile)
                        stack.append(other_coords)

    def getStateOfGrid(self):
        checking = np.array([[c.collapsed for c in row] for row in self.grid])
        return np.all(checking)


class Model:

    def __init__(self, output_size, sample_matrix):
        self.size = output_size
        self.weights, self.compatibilities, self.options = parseExampleMatrix(sample_matrix)
        self.wfc = WaveFunction(self.size, self.options, self.weights, self.compatibilities)

    def iterate(self):
        coords = self.wfc.getMinEntropyCoords()
        self.wfc.collapseCellAtCoords(coords)
        self.wfc.propagate(coords)

    def run(self):
        while not self.wfc.getStateOfGrid():
            self.iterate()
        return self.wfc.grid


def valid_dirs(cur_coords: Coordinates, matrix_size: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Returns the valid directions from `cur_co_ord` in a matrix
    of `matrix_size`. Ensures that we don't try to take step to the
    left when we are already on the left edge of the matrix.
    :param cur_coords: the current coords
    :param matrix_size: the size of the matrix
    :return: a list of possible directions
    """
    y, x = cur_coords
    dirs: list[Coordinates] = []
    height, width = matrix_size

    if y > 0:
        dirs.append(UP)
    if y < height - 1:
        dirs.append(DOWN)
    if x > 0:
        dirs.append(LEFT)
    if x < width - 1:
        dirs.append(RIGHT)

    return dirs


def parseExampleMatrix(matrix: np.ndarray) -> tuple[Weights, set[Compatibility], list[Tile]]:
    """
    Parses an example matrix. It extracts :
    1. Tile compatibilities
    2. The weight of each tile type

    :param matrix: a 2D matrix of tiles
    :return: A tuple :
        * A set of compatible tile combinations
        * A dict of weights for each tile type
    """
    compatibilities: set[Compatibility] = set()
    weights: Weights = {}
    options = list(set(matrix.flatten()))

    for y, row in enumerate(matrix):
        for x, cur_tile in enumerate(row):
            if cur_tile not in weights:
                weights[cur_tile] = 0
            weights[cur_tile] += 1

            for d in valid_dirs((y, x), matrix.shape):
                other = matrix[y+d[0], x+d[1]]
                compatibilities.add((cur_tile, other, d))  # Tile1, Tile2, Direction relative

    return weights, compatibilities, options


def render(model: Model, output_matrix, colors):
    for i in range(model.size[0]):
        row = []
        for j in range(model.size[1]):
            value = output_matrix[i, j].value
            c = colors[value]
            row.append(c + value)
        print(' '.join(row))


input_matrix = np.array([
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'C', 'C', 'L'],
    ['C', 'S', 'S', 'C'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
], dtype=Tile)

alt_input_matrix = np.array([
    ['A', 'A', 'A', 'A'],
    ['A', 'C', 'C', 'A'],
    ['C', 'B', 'B', 'C'],
    ['C', 'B', 'B', 'C'],
    ['A', 'C', 'C', 'A'],
    ['A', 'C', 'C', 'A'],
    ['A', 'A', 'A', 'A'],
], dtype=Tile)

render_colors = {
                'L': colorama.Fore.GREEN,
                'C': colorama.Fore.YELLOW,
                'S': colorama.Fore.BLUE,
                'A': colorama.Fore.CYAN,
                'B': colorama.Fore.MAGENTA}

m = Model((10, 15), input_matrix)
output = m.run()
render(m, output, render_colors)
