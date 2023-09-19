import colorama
import numpy as np
import numpy.typing as npt
import os
import time
import sys

# Types
Board = npt.NDArray[np.int64]

class GameOfLife:
    def __init__(self, rows: int = 30, cols: int = 30) -> None:
        self.rows = rows
        self.cols = cols
        self.board: Board = np.zeros((self.rows, self.cols))

    def seed(self) -> None:
        """Initialize some patterns
        """
        
        # Still lifes - don't change over time
        ## Block
        self.board[1:3, 1:3] = 1

        ## Bee-hive
        self.board[4, 2:4] = 1
        self.board[5, 1] = 1
        self.board[5, 4] = 1
        self.board[6, 2:4] = 1

        # Oscillators - change the form, but don't move
        ## Blinker
        self.board[1:4, 7] = 1

        ## Toad
        self.board[7, 7:10] = 1
        self.board[6, 8:11] = 1
        
        # Spaceships - change the form and move
        ## Glider
        self.board[2, 13] = 1
        self.board[3, 14] = 1
        self.board[1:4, 15] = 1

    def count_neighbours(self, b: Board, row: int, col: int) -> int:
        """
        Return total number of neighbours for a given cell coordinates on the board
        """
        total = (
            0 
            # row above
            + b[(row - 1) % self.rows, (col - 1) % self.cols]
            + b[(row - 1) % self.rows, col]
            + b[(row - 1) % self.rows, (col + 1) % self.cols]
            # same row
            + b[row, (col - 1) % self.cols]
            + b[row, (col + 1) % self.cols]
            # row below
            + b[(row + 1) % self.rows, (col - 1) % self.cols]
            + b[(row + 1) % self.rows, col]
            + b[(row + 1) % self.rows, (col + 1) % self.cols]
        )
        return total

    def evolve(self) -> None:
        """Update the board according to the game's rules:

        1. Any live cell with two or three live neighbours survives.
        2. Any dead cell with three live neighbours becomes a live cell.
        3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        """
        prev = self.board
        new = np.copy(self.board)

        for r in range(self.rows):
            for c in range(self.cols):
                neighbours = self.count_neighbours(b=prev, row=r, col=c)
                
                if new[r, c] == 0 and neighbours == 3:
                    new[r, c] = 1
                elif new[r, c] == 1 and ((neighbours < 2) or (neighbours > 3)):
                    new[r, c] = 0
        
        self.board = new

class StdoutRenderer:
    def __init__(self, gol: GameOfLife, speed: int = 3) -> None:
        self.gol = gol
        self.clear_cmd = "cls" if os.name == "nt" else "clear"
        self.pause = 1.0 / speed

    def clear(self) -> None:
        os.system(self.clear_cmd)

    def render(self):
        while True:
            self.clear()
            np.savetxt(sys.stdout, self.gol.board, fmt="%d")
            self.gol.evolve()
            time.sleep(self.pause)
            

if __name__ == "__main__":
    g = GameOfLife()
    g.seed()
    r = StdoutRenderer(gol=g)
    r.render()
