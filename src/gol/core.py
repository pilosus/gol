import io
import os
import sys
import time
import typing

import click
import colorama
import numpy as np
import numpy.typing as npt

# Const

COLORS = {
    "black": colorama.Fore.BLACK,
    "blue": colorama.Fore.BLUE,
    "cyan": colorama.Fore.CYAN,
    "green": colorama.Fore.GREEN,
    "magenta": colorama.Fore.MAGENTA,
    "red": colorama.Fore.RED,
    "white": colorama.Fore.WHITE,
    "yellow": colorama.Fore.YELLOW,
}

# Types
Board = npt.NDArray[np.float64]
File = io.TextIOWrapper


class GameOfLife:
    def __init__(self, seed: File, rows: int = 30, cols: int = 30) -> None:
        self.seed = seed
        self.rows = rows
        self.cols = cols
        self.board: Board = np.zeros((self.rows, self.cols))
        self.iteration = 0

    def seed_board(self) -> None:
        """Initialize the board with a seed"""
        if self.seed:
            self.board = np.loadtxt(self.seed)
            self.rows, self.cols = self.board.shape
        else:
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
        3. All other live cells die in the next generation.
        Similarly, all other dead cells stay dead.
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

        self.iteration += 1
        self.board = new


class StdoutRenderer:
    def __init__(
        self,
        gol: GameOfLife,
        speed: int,
        color_live: str,
        color_dead: str,
    ) -> None:
        self.gol = gol
        self.clear_cmd = "cls" if os.name == "nt" else "clear"
        self.speed = speed
        self.pause = 1.0 / self.speed
        self.color_live = COLORS[color_live]
        self.color_dead = COLORS[color_dead]

        np.set_printoptions(
            threshold=sys.maxsize,
            linewidth=sys.maxsize,
            formatter={"float_kind": self.format_cell},
        )

    def format_cell(self, val: np.floating[typing.Any]) -> str:
        """Format and colorize a single cell value"""
        if val > 0:
            color = self.color_live
            value = "▓"
        else:
            color = self.color_dead
            value = "░"

        return f"{color}{value} "

    def clear_screen(self) -> None:
        """Clear terminal screen"""
        os.system(self.clear_cmd)

    def output_array(self) -> None:
        """Format and print np.array"""
        for row in self.gol.board:
            for element in row:
                print(self.format_cell(element), end="")
            print(end="\n")

    def output_board(self) -> None:
        self.output_array()
        print("Board size: {}x{}".format(self.gol.rows, self.gol.cols))
        print("Evolution pace: {} iter/s".format(self.speed))
        print("Iteration:", self.gol.iteration)

    def output_exit(self) -> None:
        self.clear_screen()
        print("Exit!")

    def render(self):
        try:
            while True:
                self.clear_screen()
                self.output_board()
                self.gol.evolve()
                time.sleep(self.pause)
        except KeyboardInterrupt:
            self.output_exit()


@click.command()
@click.option(
    "--seed",
    type=click.File(mode="r", encoding=None, errors="strict", lazy=None, atomic=False),
    help="File with the board seed ('-' is a stdin)",
)
@click.option("--size", default=30, help="Board size")
@click.option("--speed", default=3, help="Iterations per second")
@click.option(
    "--color-live",
    type=click.Choice(list(COLORS), case_sensitive=False),
    default="cyan",
    help="Live cell color",
)
@click.option(
    "--color-dead",
    type=click.Choice(list(COLORS), case_sensitive=False),
    default="yellow",
    help="Dead cell color",
)
def cli(seed, size, speed, color_live, color_dead):
    g = GameOfLife(seed=seed, rows=size, cols=size)
    g.seed_board()
    r = StdoutRenderer(gol=g, speed=speed, color_live=color_live, color_dead=color_dead)
    r.render()


if __name__ == "__main__":
    cli()
