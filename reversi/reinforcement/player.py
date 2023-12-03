from reversi.sdk.player import Player
from reversi.reinforcement.trainer import GameTrainer


class MyPlayer(Player):

    def __init__(self, color):
        super().__init__(color)
        self.trainer = GameTrainer(color)

    def get_move(self, board):
        # board.display()
        action = self.trainer.get_next_move(board)
        return action
