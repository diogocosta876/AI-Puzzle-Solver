# **Cohesion Free** : Heuristic Search Methods for One Player Solitaire Games

## Game Definition 

The game [Cohesion Free](https://play.google.com/store/apps/details?id=com.NeatWits.CohesionFree&hl=en&gl=US) is a one-player game played on a pre-generated board with four different colors. The game starts with a scrambled board, and the player must slide tiles to form larger clusters of tiles of the same color. The game ends when the player wins by having only a single cluster of each color.


## Demo
![](https://github.com/diogocosta876/AI-Puzzle-Solver/blob/main/analysis/Demo.gif)


## Algorithms Used
- BFS
- DFS
- ASTAR + Heuristic
- GREEDY + Heuristic

## Setup

We used Python 3.10.10 to run the application, but it is expected to run with any version from Python 3.8-3.10 as well.
It was used [Pygame](https://www.pygame.org/news) to display the application and [plotly](https://plotly.com/python/) and [igraph](https://python.igraph.org/en/stable/) to generate the graphs of the algorithms.
You can install the necessary packages to run this application with the following command:
```bash
pip install -r requirements.txt
```

Alternatively, you can install each package manually with the following command:
```bash
pip install igraph pygame plotly
```

## Usage

In order to run the application, please run the `__main__.py` file inside the `src` folder on your IDE, or with the following command:
```bash
python3 src/__main__.py
# or
python src/__main__.py
```

## Files
The application is separated in different files for easier readability. Notice that in some files it was added a `c_*` before the name itself, this prevents any conflict between any other package of the same name you might have installed.

---

This project was made possible by:

| Name | Email |
|-|-|
| Diogo Costa | up202007770@edu.fe.up.pt |
| Fábio Sá | up202007658@edu.fe.up.pt |
| João Araújo | up202004293@edu.fe.up.pt |
