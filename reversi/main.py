#!/usr/bin/env python3
from sdk import RandomPlayer
from sdk import Game
from reversi.reinforcement.player import TrainedPlayer, ManualPlayer

import datetime

if __name__ == '__main__': 
    player1 = TrainedPlayer('X')
    player2 = RandomPlayer('O')

    has_manual_player = isinstance(player1, ManualPlayer) or isinstance(player2, ManualPlayer)
    total_timeout = 1500000 if has_manual_player else 150
    move_timeout = 60000 if has_manual_player else 6

    for _ in range(1 if has_manual_player else 100):
        game = Game(player1, player2, total_timeout, move_timeout)
        start = datetime.datetime.now()
        result = game.run()
        end = datetime.datetime.now()
        spent = (end - start).total_seconds()
        if has_manual_player:
            game.board.display()
        resultStr = result["result"]
        print(f"{resultStr}, time spent = {spent} seconds")
