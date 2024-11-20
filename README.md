# Chess AI - Utkarsh

A Chess AI project built with Python, using algorithms and techniques to enable the AI to play chess at an advanced level.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [AI Mechanics](#ai-mechanics)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements a Chess AI capable of playing against human players. It uses a combination of search algorithms and heuristics to make decisions, aiming to play optimal moves based on the current state of the board.

## Features
- Supports all standard chess moves and rules, including castling, en passant, and pawn promotion.
- AI uses **Minimax** algorithm with **Alpha-Beta Pruning** for efficient move searching.
- Implements a basic evaluation function to assess board positions and make strategic moves.

## Installation
To set up the Chess AI on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/proutkarsh3104/Chess-AI-Utkarsh.git
    cd Chess-AI-Utkarsh
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the game:
    ```bash
    python main.py
    ```

## Usage
1. Run `main.py` to start the game.
2. Use the on-screen instructions to make moves and play against the AI.

## AI Mechanics
This Chess AI is built using the Minimax algorithm with Alpha-Beta Pruning, optimizing its decision-making process by evaluating potential moves up to a specific depth. The AI calculates moves based on a scoring system that assigns values to pieces and board control.

### Future Improvements
- [ ] Improve evaluation function for better strategic play.
- [ ] Implement a GUI for easier interaction.
- [ ] Add difficulty levels by adjusting the search depth.

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and open a pull request.

1. Fork the project
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a pull request

## Video
[![Watch the video]](https://www.youtube.com/watch?v=4rCN0P21dyU)

## License
This project is licensed under the MIT License.
