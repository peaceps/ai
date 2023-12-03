import random
import json
from reversi.sdk import Board
from reversi.reinforcement.utils import get_board_state, get_opposite_player, get_abs_folder_path


class GameSampler:

    DEFAULT_VALUED = 0.

    def __init__(self, sample_config):
        self.q_sa = {'X': {}, 'O': {}}
        self.rewards = sample_config['position_rewards']
        self.alpha = sample_config['alpha']
        self.gama = sample_config['gama']
        self.epsilon = sample_config['epsilon']
        self.winner_bonus = sample_config['winner_bonus']
        self.display_steps = False

    def start_sampling(self, sampling_round, display_steps=False):
        self.q_sa = {'X': {}, 'O': {}}
        self.display_steps = display_steps
        for _ in range(sampling_round):
            self._sample_one_game()
        self._save_result(sampling_round)

    def _save_result(self, sampling_round):
        result_str = json.dumps(self.q_sa, indent=2)
        file = open(f'{get_abs_folder_path()}/files/α_{self.alpha}_γ_{self.gama}_ε_{self.epsilon}'
                    f'_bonus_{self.winner_bonus}_{sampling_round}_round_sampling.txt', 'w')
        file.write(result_str)
        file.close()

    def _sample_one_game(self):
        board = Board()
        finished, state, color = False, get_board_state(board), 'X'
        while not finished:
            finished, state, color = self._move_next(board, state, color)
        res, diff = board.get_winner()
        if diff > 0:
            winner = 'X' if res == 0 else 'O'
            bonus = diff * self.winner_bonus
            print(f'Winner is {winner} and win {diff}!')
            for i in range(4, len(board.move_history)):
                if board.move_history[i]['current_color'] == winner:
                    self._add_q_sa_value(get_board_state(board.move_history[i-1]), winner,
                                         board.move_history[i]['action'], bonus, True)
        pass

    def _move_next(self, board, state, color):
        legal_actions = list(board.get_legal_actions(color))
        self._init_q_state(state, color, legal_actions)
        taken_action = self._get_action_by_greedy_epsilon(state, color, legal_actions)
        board._move(taken_action, color, append_to_history=True)
        if self.display_steps:
            board.display()

        i, j = board.board_num(taken_action)
        r = self.rewards[i][j]

        new_state = get_board_state(board)
        factor, new_color = -1, get_opposite_player(color)
        new_legal_actions = list(board.get_legal_actions(new_color))
        if len(new_legal_actions) == 0:
            factor, new_color = 1, color
            new_legal_actions = list(board.get_legal_actions(new_color))
            if len(new_legal_actions) == 0:
                return True, '', ''
        next_gain_expectation = self._get_next_state_actions_expectation(new_state, new_color, new_legal_actions)

        current_q_sa_value = self._get_q_sa_value(state, color, taken_action)
        self._add_q_sa_value(state, color, taken_action,
                             self.alpha * (r + factor * self.gama * next_gain_expectation - current_q_sa_value))

        return False, new_state, new_color

    def _get_action_by_greedy_epsilon(self, state, color, actions):
        if len(actions) == 1:
            return actions[0]
        best_action = self._get_best_action(state, color, actions)
        r1 = random.random()
        if r1 > self.epsilon:
            taken_action = best_action
        else:
            left_actions = list(filter(lambda a: a != best_action, actions))
            taken_action = left_actions[random.randint(0, len(left_actions)-1)]
        return taken_action

    def _get_best_action(self, state, color, actions):
        max_g = -float('inf')
        best_action = ''
        for action in actions:
            g = self._get_q_sa_value(state, color, action)
            if g > max_g:
                max_g = g
                best_action = action
        return best_action

    def _get_next_state_actions_expectation(self, state, color, actions):
        best_action = self._get_best_action(state, color, actions)
        expectation = 0
        for action in actions:
            probability = 1 - self.epsilon + self.epsilon / len(actions) if action == best_action\
                else self.epsilon / len(actions)
            expectation += self._get_q_sa_value(state, color, action) * probability
        return expectation

    def _init_q_state(self, state, color, legal_actions):
        if state not in self.q_sa[color]:
            self.q_sa[color][state] = {}
            for action in legal_actions:
                self.q_sa[color][state][action] = {'count': 0, 'value': GameSampler.DEFAULT_VALUED}

    def _get_q_sa_value(self, state, color, action):
        if state not in self.q_sa[color] or action not in self.q_sa[color][state]:
            return GameSampler.DEFAULT_VALUED
        return self.q_sa[color][state][action]['value']

    def _add_q_sa_value(self, state, color, action, value, winner_bonus=False):
        action_data = self.q_sa[color][state][action]
        action_data['value'] += value
        if not winner_bonus:
            action_data['count'] += 1


if __name__ == '__main__':
    config = {
        'position_rewards': [
            [100., -35., 10., 5., 5., 10., -35., 100.],
            [-35., -35., 2., 2., 2., 2., -35., -35.],
            [10., 2., 5., 1., 1., 5., 2., 10.],
            [5., 2., 1., 2., 2., 1., 2., 5.],
            [5., 2., 1., 2., 2., 1., 2., 5.],
            [10., 2., 5., 1., 1., 5., 2., 10.],
            [-35., -35., 2., 2., 2., 2., -35., -35.],
            [100., -35., 10., 5., 5., 10., -35., 100.]
        ],
        'alpha': 0.01,
        'gama': 0.5,
        'epsilon': 0.5,
        'winner_bonus': 0.1
    }
    sampler = GameSampler(config)
    sampler.start_sampling(1)
