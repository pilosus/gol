# Conway's Game of Life

[Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
implementation inspired by Łukasz Langa's talk on [EuroPython
2023](https://github.com/ambv/gol).

Current implementation differs in that:
- it's simple, ASCII-art & terminal-based
- each evolutionary step of the game is visible in the terminal history
- a seed (initial board) can be passed in as a user input

That makes the game fun and easy to experiment with.

![screencast](https://blog.pilosus.org/images/gol.gif)

# Usage

1. Clone the repo

2. Install the package locally in [editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) mode

```bash
$ make install-package
```

3. Run the Game of Life with the sensible defaults

```bash
$ gol
```

4. Use command line options to adjust the gameplay as needed

```bash
$ gol --seed docs/seeds/basic.txt \
    --speed=10 \
    --color-live=green \
    --color-dead=white
```

5. Get help

```
$ gol --help
```

# Seed from a file

A board seed can be provided as a file using `--seed path/to/a/file`
CLI option. The seed file must be a text file representing 2D
array. Dead cells represented by value `0`, live cells by the value
`1`. Elements in a row separated by a single space character.

When the seed file is passed in, `--size` option is ignored and
adjusted to the share of the seed array automatically.

See an [example seed file](https://github.com/pilosus/gol/blob/main/docs/seeds/basic.txt)
in the repo.
