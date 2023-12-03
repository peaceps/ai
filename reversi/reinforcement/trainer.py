import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reversi.sdk import Board
from reversi.reinforcement.utils import (get_board_state, state_num, action_num, num_action,
                                         move_and_get_next_color, get_abs_folder_path)


folder_path = get_abs_folder_path()
model_path = {'X': f'{folder_path}/files/model_X.pt', 'O': f'{folder_path}/files/model_O.pt'}
q_sa_file_path = f'{folder_path}/files/α_0.01_γ_0.5_ε_0.5_bonus_0.1_1_round_sampling.txt'


class GameDataLoader:

    @staticmethod
    def load_input_output_data(file):
        q_sa = json.load(file)
        inputs = {}
        outputs = {}
        for color in ['X', 'O']:
            color_inputs = []
            color_outputs = []
            for state in q_sa[color].keys():
                input_tensor, output_tensor = GameDataLoader.convert_state_to_tensor(state, q_sa[color][state])
                color_inputs.append(input_tensor)
                color_outputs.append(output_tensor)

            inputs[color] = torch.stack(color_inputs)
            outputs[color] = torch.stack(color_outputs)

        return inputs, outputs

    @staticmethod
    def convert_state_to_tensor(state, actions):
        num_state = list(map(state_num, state))
        state_tensor = torch.tensor(num_state).reshape(8, -1)
        action_tensor = torch.zeros(state_tensor.size())
        output_tensor = torch.zeros(state_tensor.size()) + 100

        is_dict = isinstance(actions, dict)
        action_strs = actions if not is_dict else list(actions.keys())
        for action in action_strs:
            x, y = action_num(action)
            action_tensor[x][y] += 1
            if is_dict:
                output_tensor[x][y] += actions[action]['value']
        input_tensor = torch.stack([state_tensor, action_tensor])
        return input_tensor, output_tensor


class GameNetwork(nn.Module):

    def __init__(self):
        super(GameNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)

    def forward(self, x):
        actions = x[1].clone().detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(8, -1)
        x = x * actions

        return x


class GameTrainer:

    def __init__(self, color):
        self.color = color
        self.net = GameNetwork()
        model_exist = os.path.isfile(model_path[color])
        if model_exist:
            self.net.load_state_dict(torch.load(model_path[color]))
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        if not model_exist:
            inputs, outputs = GameDataLoader().load_input_output_data(open(q_sa_file_path, 'r'))
            self.train(inputs[color], outputs[color])

    def train(self, inputs, targets):
        self.net.train()
        start = time.time()
        for epoch in range(2):
            running_loss = 0.0
            for i, input_data in enumerate(inputs, 0):
                self.optimizer.zero_grad()
                output = self.net(input_data)
                loss = self.criterion(output, targets[i])
                loss.backward()
                self.optimizer.step()

                # 打印统计信息
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training! Total cost time: ', time.time() - start)
        torch.save(self.net.state_dict(), model_path[self.color])

    def get_next_move(self, board):
        self.net.eval()
        state = get_board_state(board)
        actions = board.get_legal_actions(self.color)
        inputs, _ = GameDataLoader.convert_state_to_tensor(state, actions)
        outputs = self.net(inputs) + inputs[1] * 100
        max_0 = outputs.max(0)
        max_v = max_0[0].max(0)[1].item()
        x, y = max_0[1][max_v].item(), max_v
        return num_action(x, y)


def validate_moves(trainer):
    board = Board()
    color = 'X'
    color = move_and_get_next_color(board, 'F5', color)
    color = move_and_get_next_color(board, 'F6', color)
    board.display()
    next_move = trainer[color].get_next_move(board)
    print(f'next color is {color} and move is {next_move}')


if __name__ == '__main__':
    trainer = {'X': GameTrainer('X'), 'O': GameTrainer('O')}
    validate_moves(trainer)
