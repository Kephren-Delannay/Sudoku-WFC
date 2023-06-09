import random
import numpy as np


class Tile:
    options = []
    collapsed = False
    value = None
    position = ()

    def __init__(self, y, x):
        self.options = [i for i in range(1, 10)] # set options for a cell from 1 to 9
        self.collapsed = False
        self.value = None
        self.position = (y, x)

    def setValue(self, v):
        self.value = v
        self.options = []
        self.collapsed = True

    def getEnthropy(self):
        return len(self.options)

    def setOptions(self, selectedOption):
        """
        Updates the available options for a given cell based on the selected option from a neighbouring cell \n
        For this example we simply remove the selected option from the available ones but in more complex grids, we should verify the validity of each option based on the selected one
        :param selectedOption: the option that was selected on the collapse
        :return: None
        """
        if selectedOption in self.options:
            self.options.remove(selectedOption)

    def collapse(self):
        """
        collapse the selected cell to a single value at random (this could take into account the biais of the input)
        """
        v = random.choice(self.options)
        self.setValue(v)

    def __str__(self):
        return f'tile {self.position} \n' \
               f'options : {[i for i in self.options]} \n' \
               f'value : {self.value} \n' \
               f'collapsed : {self.collapsed} \n'

    def __le__(self, other):
        return self.getEnthropy() < other.getEnthropy()

    def __eq__(self, other):
        return self.getEnthropy() == other.getEnthropy()

    def __gt__(self, other):
        return self.getEnthropy() > other.getEnthropy()


class Sudoku:
    grid = np.empty(shape=(9, 9), dtype=Tile)

    def __init__(self):
        for y in range(9):
            for x in range(9):
                self.grid[y, x] = Tile(y, x)

    def __repr__(self):
        display = f''
        for y in range(9):
            for x in range(9):
                # display += (" " + str(target[y, x].value) + " ")
                display += f'{str(self.grid[y, x].value):>4} '
            display += "\n"
        return display

    def validateGrid(self):
        for i in range(9):
            try:
                row_sum = sum([t.value for t in self.grid[i, :]])
                col_sum = sum([t.value for t in self.grid[:, i]])
            except TypeError:
                return False
            if row_sum != 45 or col_sum != 45:
                return False
        return True


class UnfinishedError(Exception):
    pass


class FinishedError(Exception):
    pass


def getMinEnthropyCell(sudoku):
    """
    Return the cell with the least enthropy in the grid
    :return: a cell
    """
    # for now it wil always get the first option of the sorted list
    # TODO: add randomness

    collapsed_filter = np.array([t.collapsed for t in sudoku.grid.ravel()])

    # just getting the non collapsed values
    l = list(sudoku.grid.ravel()[~collapsed_filter])

    # return the first element of the sorted list
    return sorted(l)[0]


def propagateChanges(sudoku, cell):
    """
    Propagate for 1 cycle only the changes in the grid to neighbouring cells
    :param cell: the collapsed cell
    :return: None
    """
    y,x = cell.position

    for i in range(9):
        # propagate on the row
        sudoku.grid[y, i].setOptions(cell.value)

        # propagate on the column
        sudoku.grid[i, x].setOptions(cell.value)

    # propagate around
    for y_offset in range(-1, 2, 1):
        for x_offset in range(-1, 2, 1):
            new_y = np.clip(y + y_offset, 0, 8)
            new_x = np.clip(x + x_offset, 0, 8)
            if new_y == y and new_x == x: # selected cell
                continue
            sudoku.grid[new_y, new_x].setOptions(cell.value)


def algorithm(sudoku):
    try:
        selected_cell = getMinEnthropyCell(sudoku)
    except IndexError:
        raise FinishedError()
    try:
        selected_cell.collapse()
    except IndexError:
        raise UnfinishedError("Cannot finish")
    propagateChanges(sudoku, selected_cell)
    print(sudoku)


if __name__ == '__main__':
    sudoku_grid = Sudoku()
    i = 0
    attempts = 1
    while True:
        try:
            algorithm(sudoku_grid)
            i += 1
        except UnfinishedError:
            print(f'Failed in {i} steps')
            print('\n'*10)
            sudoku_grid = Sudoku()
            i = 0
            attempts += 1
        except FinishedError:
            print(f'Success in {attempts} attempts')
            print(sudoku_grid.validateGrid())
            break
