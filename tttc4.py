import math
import os
import time
import sys
import random
import pickle

import numpy as np
import pygame
import pygame_menu
from random import sample
import copy

CELL = (255, 191, 223)
WALL = (255, 255, 255)
SHADOW = (99, 0, 49)
TEXT = (207, 0, 88)
START = (162, 191, 67)
END = (250, 74, 47)
BUTTON = (255, 219, 237)

sys.setrecursionlimit(3000)
pygame.init()
pygame.display.set_caption('Assignment')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def easy(grid):
    legal_moves = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == -1:
                legal_moves.append([r,c])
    return sample(legal_moves, 1)[0]

def medium_ttt(grid, player):
    opponent = 1 - player
    grid_flat = np.array(grid).flatten()
    player_wins = [player, player, -1], [player, -1, player], [-1, player, player]
    opponent_wins = [opponent, opponent, -1], [opponent, -1, opponent], [-1, opponent, opponent]
    lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    coords = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    # Check winning moves first
    for line in lines:
        l = [grid_flat[line[0]], grid_flat[line[1]], grid_flat[line[2]]]
        if l in player_wins:
            for position in line:
                if grid_flat[position] == -1:
                    return coords[position]
    # Check saving moves second
    for line in lines:
        l = [grid_flat[line[0]], grid_flat[line[1]], grid_flat[line[2]]]
        if l in opponent_wins:
            for position in line:
                if grid_flat[position] == -1:
                    return coords[position]
    legal_moves = [i for i in range(9) if grid_flat[i] == -1]
    try:
        return coords[sample(legal_moves, 1)[0]]
    except:
        return None


def medium_c4(grid, player):
    opponent = 1 - player
    player_wins = [[player, player, player, -1], [player, player, -1, player], [player, -1, player, player],
                   [-1, player, player, player]]
    opponent_wins = [[opponent, opponent, opponent, -1], [opponent, opponent, -1, opponent],
                     [opponent, -1, opponent, opponent], [-1, opponent, opponent, opponent]]
    for win_set in [player_wins, opponent_wins]:
        for i in range(6):
            for j in range(4):
                positions = [[i, j], [i, j + 1], [i, j + 2], [i, j + 3]]
                if [grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3]] in win_set:
                    for m, move in enumerate([grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3]]):
                        if move == -1:
                            return positions[m]
        for j in range(7):
            for i in range(3):
                positions = [[i, j], [i + 1, j], [i + 2, j], [i + 3, j]]
                if [grid[i][j], grid[i + 1][j], grid[i + 2][j], grid[i + 3][j]] in win_set:
                    for m, move in enumerate([grid[i][j], grid[i + 1][j], grid[i + 2][j], grid[i + 3][j]]):
                        if move == -1:
                            return positions[m]
        for i in range(3):
            for j in range(4):
                positions = [[i, j], [i + 1, j + 1], [i + 2, j + 2], [i + 3, j + 3]]
                if [grid[i][j], grid[i + 1][j + 1], grid[i + 2][j + 2], grid[i + 3][j + 3]] in win_set:
                    for m, move in enumerate([grid[i][j], grid[i + 1][j + 1], grid[i + 2][j + 2], grid[i + 3][j + 3]]):
                        if move == -1:
                            return positions[m]
                positions = [[i, j + 3], [i + 1, j + 2], [i + 2, j + 1], [i + 3, j]]
                if [grid[i][j + 3], grid[i + 1][j + 2], grid[i + 2][j + 1], grid[i + 3][j]] in win_set:
                    for m, move in enumerate([grid[i][j + 3], grid[i + 1][j + 2], grid[i + 2][j + 1], grid[i + 3][j]]):
                        if move == -1:
                            return positions[m]

    legal_moves = []
    for j in range(7):
        for i in range(6):
            if grid[i][j] == -1:
                legal_moves.append([i, j])
    try:
        return sample(legal_moves, 1)[0]
    except:
        return None


def minimax_ttt(grid, turn, player, opponent, alpha, beta):
    legal_moves = []
    for j in range(3):
        for i in range(3):
            if grid[i][j] == -1:
                legal_moves.append([i, j])
    best_move = [-math.inf, None, None] if turn == player else [math.inf, None, None]
    if len(legal_moves) == 0:
        return evaluate_ttt(grid, player), None, None
    for move in legal_moves:
        grid_new = np.copy(grid)
        r, c = move
        grid_new[r][c] = turn
        payoff = evaluate_ttt(grid_new, player)
        if payoff == None:
            payoff = minimax_ttt(grid_new, 1 - turn, player, opponent, alpha, beta)[0]
        if turn == player:
            if payoff > best_move[0]:
                best_move = [payoff, r, c]
            alpha = max(alpha, best_move[0])
        else:
            if payoff < best_move[0]:
                best_move = [payoff, r, c]
            beta = min(beta, best_move[0])
        if beta <= alpha:
            break
    return best_move


def evaluate_ttt(grid, player):
    grid_flat = np.array(grid).flatten()
    opponent = 1 - player
    if grid[2][0] == grid[1][1] == grid[0][2]:
        if grid[1][1] == player:
            return 1
        if grid[1][1] == opponent:
            return -1
    if grid[0][0] == grid[1][1] == grid[2][2]:
        if grid[1][1] in [0, 1]:
            if grid[1][1] == player:
                return 1
            if grid[1][1] == opponent:
                return -1
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2]:
            if grid[i][1] == player:
                return 1
            if grid[i][1] == opponent:
                return -1
        elif grid[0][i] == grid[1][i] == grid[2][i]:
            if grid[1][i] == player:
                return 1
            if grid[1][i] == opponent:
                return -1
    if -1 not in grid_flat:
        return 0
    return None


def minimax_c4(grid, turn, player, opponent, alpha, beta, depth):
    legal_moves = []
    for i in range(6):
        for j in range(7):
            if grid[i][j] == -1:
                legal_moves.append([i, j])
    best_move = [-math.inf, None, None] if turn == player else [math.inf, None, None]
    payoff = evaluate_c4(grid, player)
    if depth == 0 or abs(payoff) > 100 or len(legal_moves) == 0:
        return payoff, None, None
    for move in legal_moves:
        grid_new = np.copy(grid)
        r, c = move
        grid_new[r][c] = turn
        if r > 0:
            grid_new[r - 1][c] = -1
        payoff = minimax_c4(grid_new, 1 - turn, player, opponent, alpha, beta, depth - 1)[0]
        if turn == player:
            if payoff > best_move[0]:
                best_move = [payoff, r, c]
            alpha = max(alpha, best_move[0])
        else:
            if payoff < best_move[0]:
                best_move = [payoff, r, c]
            beta = min(beta, best_move[0])
        if beta <= alpha:
            break
    return best_move


def evaluate_c4(grid, player):
    grid_flat = np.array(grid).flatten()
    opponent = 1 - player
    score = 0
    for i in range(6):
        for j in range(4):
            score = score + scoring([grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3]], player)
    for j in range(7):
        for i in range(3):
            score = score + scoring([grid[i][j], grid[i + 1][j], grid[i + 2][j], grid[i + 3][j]], player)
    for i in range(3):
        for j in range(4):
            score = score + scoring([grid[i][j], grid[i + 1][j + 1], grid[i + 2][j + 2], grid[i + 3][j + 3]], player)
            score = score + scoring([grid[i][j + 3], grid[i + 1][j + 2], grid[i + 2][j + 1], grid[i + 3][j]], player)
    if -1 not in grid_flat:
        return 0
    return score

def scoring(points, player):
    count_player = sum(1 for p in points if p == player)
    count_opponent = sum(1 for p in points if p == 1 - player)
    if count_player == 4:
        return 300
    if count_opponent == 4:
        return -400
    if count_opponent == 0:
        return count_player
    if count_player == 0:
        return -count_opponent
    return 0

def evaluate_c4_q(grid, player):
    grid_flat = np.array(grid).flatten()
    opponent = 1 - player
    score = 0
    for i in range(6):
        for j in range(4):
            score = scoring([grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3]], player)
            if score == 300:
                return 1
            if score == -400:
                return -1
    for j in range(7):
        for i in range(3):
            score = scoring([grid[i][j], grid[i + 1][j], grid[i + 2][j], grid[i + 3][j]], player)
            if score == 300:
                return 1
            if score == -400:
                return -1
    for i in range(3):
        for j in range(4):
            score = scoring([grid[i][j], grid[i + 1][j + 1], grid[i + 2][j + 2], grid[i + 3][j + 3]], player)
            if score == 300:
                return 1
            if score == -400:
                return -1
            score = scoring([grid[i][j + 3], grid[i + 1][j + 2], grid[i + 2][j + 1], grid[i + 3][j]], player)
            if score == 300:
                return 1
            if score == -400:
                return -1
    if -1 not in grid_flat:
        return 0
    return None


def q_learning_train_ttt(player, alpha=0.2, gamma=0.9, epsilon=0.2):
    moves = [[i, j] for i in range(3) for j in range(3)]
    file = 'q' + str(player) + 'ttt.pickle'
    try:
        with open(file, 'rb') as f:
            q_table = pickle.load(f)
    except:
        q_table = {}
    for game in range(1000):
        grid = [[-1 for _ in range(3)] for _ in range(3)]
        legal_moves = [i for i in range(9)]
        # If the opponent is X, it makes the first move
        if (player == 1):
            m_opponent = sample(legal_moves, 1)[0]
            grid[moves[m_opponent][0]][moves[m_opponent][1]] = 1 - player
            legal_moves.pop(legal_moves.index(m_opponent))
        q_move_ttt(q_table, grid, legal_moves, player, alpha, gamma, epsilon)
    with open(file, 'wb') as f:
        pickle.dump(q_table, f)


def q_move_ttt(q_table, grid, legal_moves, player, alpha, gamma, epsilon):
    state = grid_to_state(grid)
    moves = [[i, j] for i in range(3) for j in range(3)]
    reward = evaluate_ttt(grid, player)
    # Return the final state value if it is a leaf
    if reward is not None:
        return reward
    else:
        if state in q_table:
            # Try to pick the best move for the current state, pick a random move and add it if there are no other moves in the state
            try:
                m = max(q_table[state], key=q_table[state].get)
            except:
                m = sample(legal_moves, 1)[0]
        else:
            # Add state to the dictionary if it doesn't exist and pick a random move
            q_table[state] = {}
            m = sample(legal_moves, 1)[0]
        # With probability epsilon, pick a random move regardless
        if random.randint(1, 10) <= epsilon * 10:
            m = sample(legal_moves, 1)[0]
        # Adding the move to the table if it doesn't exist
        if m not in q_table[state]:
            q_table[state][m] = 0
        # Make the move
        new_grid = np.copy(grid)
        new_grid[moves[m][0]][moves[m][1]] = player
        legal_moves.pop(legal_moves.index(m))
        # If it's not a leaf after the new move, opponent moves too
        reward = evaluate_ttt(new_grid, player)
        if reward is None:
            # [r,c] = q_learning_ttt(new_grid, 1-player)
            # [p,r,c] = minimax_ttt(new_grid,1-player,1-player,player,-math.inf,math.inf)
            [r, c] = medium_ttt(new_grid, 1 - player)
            m_opponent = moves.index([r, c])
            new_grid[r][c] = 1 - player
            legal_moves.pop(legal_moves.index(m_opponent))
        # Updating q_table
        q_table[state][m] += alpha * (
                gamma * q_move_ttt(q_table, new_grid, legal_moves, player, alpha, gamma, epsilon) - q_table[state][
            m])
    # Returning the best value for the parent q-value
    return max(q_table[state].values())


def q_learning_train_c4(player, alpha=0.2, gamma=0.9, epsilon=0.2, opponent = 'Random'):
    file = 'q' + str(player) + 'c4.pickle'
    file = 'q' + str(1-player) + 'c4.pickle'
    try:
        with open(file, 'rb') as f:
            q_table = pickle.load(f)
    except:
        q_table = {}
    try:
        with open(file, 'rb') as f:
            q_table_opponent = pickle.load(f)
    except:
        q_table_opponent = {}
    for game in range(10000):
        grid = [[-2 for _ in range(7)] for _ in range(5)]
        grid.append([-1 for _ in range(7)])
        legal_moves = [i for i in range(7)]
        # If the opponent is X, it makes the first move
        if (player == 1):
            m_opponent = sample(legal_moves, 1)[0]
            grid[5][m_opponent] = 0
            grid[4][m_opponent] = -1
        q_move_c4(q_table, q_table_opponent, grid, legal_moves, player, alpha, gamma, epsilon, opponent)
    with open(file, 'wb') as f:
        pickle.dump(q_table, f)


def q_move_c4(q_table, q_table_opponent, grid, legal_moves, player, alpha, gamma, epsilon, opponent = 'Random'):
    state = grid_to_state(grid)
    reward = evaluate_c4_q(grid, player)
    moves = copy.deepcopy(legal_moves)
    # Return the final state value if it is a leaf
    if reward is not None:
        return reward
    else:
        if state in q_table:
            # Try to pick the best move for the current state, pick a random move and add it if there are no other moves in the state
            try:
                m = max(q_table[state], key=q_table[state].get)
            except:
                m = sample(moves, 1)[0]
        else:
            # Add state to the dictionary if it doesn't exist and pick a random move
            q_table[state] = {}
            m = sample(moves, 1)[0]
        # With probability epsilon, pick a random move regardless
        if random.randint(1, 10) <= epsilon * 10:
            m = sample(moves, 1)[0]
        # Adding the move to the table if it doesn't exist
        if m not in q_table[state]:
            q_table[state][m] = 0
        # Make the move
        new_grid = np.copy(grid)
        for row in range(5, -1, -1):
            if new_grid[row][m] == -1:
                new_grid[row][m] = player
                if row > 0:
                    new_grid[row - 1][m] = -1
                else:
                    moves.pop(moves.index(m))
                break

        # If it's not a leaf after the new move, opponent moves too
        reward = evaluate_c4_q(new_grid, player)
        if reward is None:
            #c = sample(legal_moves, 1)[0]
            c = medium_c4(new_grid, 1 - player)[1]  if opponent == 'Medium' else q_learning_c4(new_grid, 1 - player, q_table_opponent) if opponent == 'QLearning' else sample(moves, 1)[0]
            # [p,r,c] = minimax_c4(new_grid,1-player,1-player,player,-math.inf,math.inf,5)
            #[r, c] = easy_c4(new_grid, 1 - player)
            for row in range(5, -1, -1):
                if new_grid[row][c] == -1:
                    new_grid[row][c] = 1 - player
                    if row > 0:
                        new_grid[row - 1][c] = -1
                    else:
                        moves.pop(moves.index(c))
                    break
        # Updating q_table
        q_table[state][m] += alpha * (
                gamma * q_move_c4(q_table, q_table_opponent, new_grid, moves, player, alpha, gamma, epsilon, opponent) - q_table[state][m])
    # Returning the best value for the parent q-value
    return max(q_table[state].values())


def grid_to_state(grid):
    state = ""
    for row in grid:
        for cell in row:
            state += str(cell)
    return state


def q_learning_ttt(grid, player):
    moves = [[i, j] for i in range(3) for j in range(3)]
    file = 'q' + str(player) + 'ttt.pickle'
    try:
        with open(file, 'rb') as f:
            q_table = pickle.load(f)
    except:
        print("This player is untrained.")
    state = grid_to_state(grid)
    try:
        best_move = max(q_table[state], key=q_table[state].get)
        return moves[best_move]
    except:
        # print("This game state has not been encountered yet. A random move will be played")
        legal_moves = []
        for i in range(3):
            for j in range(3):
                if grid[i][j] == -1:
                    legal_moves.append([i, j])
        return sample(legal_moves, 1)[0]


def q_learning_c4(grid, player, q_table = None):
    moves = [[i, j] for i in range(3) for j in range(3)]
    file = 'q' + str(player) + 'c4.pickle'
    if q_table == None:
        try:
            with open(file, 'rb') as f:
                q_table = pickle.load(f)
        except:
            print("This player is untrained.")
    state = grid_to_state(grid)
    try:
        best_move = max(q_table[state], key=q_table[state].get)
        return best_move
    except:
        # print("This game state has not been encountered yet. A random move will be played")
        legal_moves = [i for i in range(7)]
        for j in range(7):
            if grid[0][j] in [0, 1]:
                legal_moves.pop(legal_moves.index(j))
        return sample(legal_moves, 1)[0]


def draw_tic_tac_toe(screen, Y):
    screen.fill(CELL)
    for i in range(1, 3):
        pygame.draw.line(screen, BUTTON, (10, i * Y / 3), (Y - 10, i * Y / 3), 3)
        pygame.draw.line(screen, BUTTON, (i * Y / 3, 10), (i * Y / 3, Y - 10), 3)


def draw_connect_four(screen, Y):
    screen.fill(CELL)
    # pygame.draw.rect(screen,BUTTON,(0,0,Y*7,Y*7))
    for i in range(7):
        for j in range(1, 7):
            pygame.draw.circle(screen, BUTTON, (Y * i + Y / 2, Y * j + Y / 2), Y / 2.5)


def draw_win_ttt(screen, Y, player, sX, sY, eX, eY):
    winner = ['X won!', 'O won!', 'Draw!']
    colour = [SHADOW, WALL, CELL]
    pygame.draw.line(screen, colour[player], (sX, sY), (eX, eY), Y // 10)
    s = pygame.Surface((Y * 3, Y * 3))
    s.set_alpha(160)
    s.fill(CELL)
    screen.blit(s, (0, 0))
    font = pygame.font.Font(None, 8 + Y // 2)
    text = font.render(winner[player], True, TEXT)
    textRect = text.get_rect()
    textRect.center = (Y * 3 // 2, Y * 3 // 2)
    screen.blit(text, textRect)


def draw_win_c4(screen, Y, player, p1, p2, p3, p4):
    winner = ['Player 1 won!', 'Player 2 won!', 'Draw!']
    colour = [SHADOW, WALL, CELL]
    try:
        for p in [p1, p2, p3, p4]:
            pygame.draw.ellipse(screen, TEXT, (Y * p[1], Y * p[0] + Y, Y, Y), 5)
    except:
        print('Skipping')
    pygame.display.flip()
    s = pygame.Surface((Y * 7, Y * 7))
    s.set_alpha(160)
    s.fill(CELL)
    screen.blit(s, (0, 0))
    font = pygame.font.Font(None, 8 + Y)
    text = font.render(winner[player], True, TEXT)
    textRect = text.get_rect()
    textRect.center = (Y * 7 // 2, Y * 7 // 2)
    screen.blit(text, textRect)


def draw_turn_ttt(screen, Y, i, j, player):
    X_0 = [pygame.image.load(os.path.join('X.svg')), pygame.image.load(os.path.join('0.svg'))]
    X_0 = [pygame.transform.smoothscale(picture, (Y, Y)) for picture in X_0]
    screen.blit(X_0[player], (i * Y, j * Y, Y, Y))


def draw_turn_c4(screen, Y, i, j, player):
    colour = [SHADOW, WALL]
    pygame.draw.circle(screen, colour[player], (Y * i + Y / 2, Y * j + Y / 2), Y / 2.5)


def runMenu(screen, mytheme):
    player = [0, 0]  # player 1: X, player 2: 0; human = 0, dummy = 1, minmax = 2, RI = 3
    delay = '0'
    games = '100'
    buttonSize, labelSize, lineSize = max(1, screen.get_size()[1] // 20 - 6), max(1,
                                                                                  screen.get_size()[1] // 25 - 4), max(
        1, screen.get_size()[1] // 60 - 4)

    def update_delay(time):
        nonlocal delay
        delay = str(time)

    def update_games(number):
        nonlocal games
        games = str(number)

    def update_player1(name, choice):
        nonlocal player
        player[0] = choice

    def update_player2(name, choice):
        nonlocal player
        player[1] = choice

    def setY(size):
        if size.isdigit() and int(size) >= 300:
            screenY = int(size)
            screenX = 1.4 * screenY
            screen = pygame.display.set_mode((screenX, screenY))
            screen.fill(WALL)
            runMenu(screen, mytheme)

    run_ttt = lambda: print("The number of games must be a positive integer.") if not games.isdigit() or int(
        games) < 0 else print("The delay must be a positive rational number") if not isfloat(delay) or float(
        delay) < 0 else run_tic_tac_toe(screen, player, float(delay), int(games))
    run_c4 = lambda: print("The number of games must be a positive integer.") if not games.isdigit() or int(
        games) < 0 else print("The delay must be a positive rational number") if not isfloat(delay) or float(
        delay) < 0 else run_connect_four(screen, player, float(delay), int(games))

    main_menu = pygame_menu.Menu('Main Menu', screen.get_size()[0], screen.get_size()[1], theme=mytheme)
    main_menu.add.dropselect('Player 1 :', [('Human', 0), ('Easy', 1), ('Medium', 2),('Hard - Minimax', 3), ('Hard - Q-Learning', 4)],
                             font_size=buttonSize, onchange=update_player1, default=0, selection_box_bgcolor=CELL,
                             selection_box_border_color=WALL, selection_option_font_color=WALL,
                             selection_option_selected_font_color=TEXT,
                             selection_option_selected_bgcolor=BUTTON).set_border(2, SHADOW,
                                                                                  position=('position-south',
                                                                                            'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.dropselect('Player 2 :', [('Human', 0), ('Easy', 1), ('Medium', 2),('Hard - Minimax', 3), ('Hard - Q-Learning', 4)],
                             font_size=buttonSize, onchange=update_player2, default=0, selection_box_bgcolor=CELL,
                             selection_box_border_color=WALL, selection_option_font_color=WALL,
                             selection_option_selected_font_color=TEXT,
                             selection_option_selected_bgcolor=BUTTON).set_border(2, SHADOW,
                                                                                  position=('position-south',
                                                                                            'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()

    main_menu.add.label('', font_size=lineSize)
    main_menu.add.label('Settings:', font_size=labelSize)
    main_menu.add.text_input('Computer delay: ', default=0, font_size=buttonSize, onchange=update_delay).set_border(2,
                                                                                                                      SHADOW,
                                                                                                                      position=(
                                                                                                                          'position-south',
                                                                                                                          'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.text_input('Games: ', default=100, font_size=buttonSize, onchange=update_games).set_border(2,
                                                                                                           SHADOW,
                                                                                                           position=(
                                                                                                               'position-south',
                                                                                                               'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.text_input('Window height: ', default=str(screen.get_size()[1]), font_size=buttonSize,
                             onchange=setY).set_border(2, SHADOW,
                                                       position=('position-south', 'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.label('', font_size=lineSize)
    main_menu.add.button('Play Tic Tac Toe', run_ttt, font_size=buttonSize).set_border(2, SHADOW, position=(
        'position-south', 'position-east')).set_margin(max(3, screen.get_size()[0] / 60 + 10),
                                                       max(3, screen.get_size()[0] / 40 - 5))
    main_menu.add.button('Play Connect-four', run_c4, font_size=buttonSize).set_border(2, SHADOW, position=(
        'position-south', 'position-east')).set_margin(max(3, screen.get_size()[0] / 60 + 10),
                                                       max(3, screen.get_size()[0] / 40 - 5))
    main_menu.add.button('Quit', pygame_menu.events.EXIT, font_size=buttonSize).set_border(2, SHADOW, position=(
        'position-south', 'position-east')).set_margin(max(3, screen.get_size()[0] / 60 + 10),
                                                       max(3, screen.get_size()[0] / 40 - 5))
    main_menu.enable()
    running = True
    while running:
        events = pygame.event.get()
        if main_menu.is_enabled():
            main_menu.update(events)
            main_menu.draw(screen)
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break
        pygame.display.flip()


def run_tic_tac_toe(screen, player, delay, games):
    grid = [[-1 for _ in range(3)] for _ in range(3)]
    turn = 0
    screenY = screen.get_size()[1]
    Y = screenY // 3
    screenX = screen.get_size()[0]
    end = False
    game = 1
    outcomes = [0, 0, 0]

    def run_clear(screen):
        nonlocal grid
        nonlocal turn
        nonlocal end
        turn = 0
        draw_tic_tac_toe(screen, screenY)
        grid = [[-1 for _ in range(3)] for _ in range(3)]
        end = False

    def draw(screen, i, j):
        nonlocal grid
        nonlocal turn
        if grid[i][j] == -1:
            grid[i][j] = turn
            draw_turn_ttt(screen, Y, i, j, turn)
            if check_win() == 'Continue':
                turn = 1 - turn

    def check_win():
        nonlocal grid
        if grid[2][0] == grid[1][1] == grid[0][2]:
            if grid[1][1] in [0, 1]:
                return run_win(grid[1][1], 2, 0, 0, 2)
        if grid[0][0] == grid[1][1] == grid[2][2]:
            if grid[1][1] in [0, 1]:
                return run_win(grid[1][1], 0, 0, 2, 2)
        else:
            for i in range(3):
                if grid[i][0] == grid[i][1] == grid[i][2]:
                    if grid[i][1] in [0, 1]:
                        return run_win(grid[i][1], i, 0, i, 2)
                elif grid[0][i] == grid[1][i] == grid[2][i]:
                    if grid[1][i] in [0, 1]:
                        return run_win(grid[1][i], 0, i, 2, i)
        if -1 not in np.array(grid).flatten():
            return run_win(2, -1, -1, -1, -1)
        else:
            return 'Continue'

    def run_win(player, sX, sY, eX, eY):
        nonlocal turn
        nonlocal grid
        nonlocal end
        nonlocal game
        nonlocal outcomes
        outcomes[player] = outcomes[player] + 1
        game = game + 1
        end = True
        sX, sY, eX, eY = np.array([sX, sY, eX, eY]) * Y + Y / 2

        for i in range(3):
            for j in range(3):
                if grid[i][j] == -1:
                    grid[i][j] = 2
        draw_win_ttt(screen, Y, player, sX, sY, eX, eY)
        pygame.display.flip()
        time.sleep(delay)
        pygame.display.flip()
        if game <= games:
            run_clear(screen)
        else:
            print(outcomes)

    buttonSize, labelSize, lineSize = max(3, screen.get_size()[1] // 20 - 6), max(3,
                                                                                  screen.get_size()[1] // 25 - 4), max(
        3, screen.get_size()[1] // 60 - 4)

    draw_tic_tac_toe(screen, screenY)

    menu = pygame_menu.Menu("", screenX - screenY, screenY, False, position=(100, 0), theme=mytheme)

    grid_buttons = [[pygame.Rect(0, 0, screenY / 3, screenY / 3), pygame.Rect(0, screenY / 3, screenY / 3, screenY / 3),
                     pygame.Rect(0, 2 * screenY / 3, screenY / 3, screenY / 3)],
                    [pygame.Rect(screenY / 3, 0, screenY / 3, screenY / 3),
                     pygame.Rect(screenY / 3, screenY / 3, screenY / 3, screenY / 3),
                     pygame.Rect(screenY / 3, 2 * screenY / 3, screenY / 3, screenY / 3)],
                    [pygame.Rect(2 * screenY / 3, 0, screenY / 3, screenY / 3),
                     pygame.Rect(2 * screenY / 3, screenY / 3, screenY / 3, screenY / 3),
                     pygame.Rect(2 * screenY / 3, 2 * screenY / 3, screenY / 3, screenY / 3)]]

    clearButton = menu.add.button('Clear', lambda: run_clear(screen), font_size=labelSize)
    menuButton = menu.add.button('Main Menu', lambda: runMenu(screen, mytheme), font_size=labelSize)
    quizButton = menu.add.button('Quit', pygame_menu.events.EXIT, font_size=labelSize)

    buttons = [clearButton, menuButton, quizButton]
    for button in buttons:
        button.set_margin(max(1, screenY / 60 + 10), max(1, screenY / 40 - 5))
        button.set_border(2, SHADOW, position=('position-south', 'position-east'))
    running = True
    while running:
        events = pygame.event.get()
        if menu.is_enabled():
            menu.update(events)
            menu.draw(screen)
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(3):
                    for j in range(3):
                        if grid_buttons[i][j].collidepoint(event.pos):
                            draw(screen, i, j)
        if not end:
            if player[turn] == 1:
                pygame.display.flip()
                r, c = easy(grid)
                time.sleep(delay)
                draw(screen, r, c)
            if player[turn] == 2:
                pygame.display.flip()
                r, c = medium_ttt(grid, turn)
                time.sleep(delay)
                draw(screen, r, c)
            if player[turn] == 3:
                pygame.display.flip()
                p, r, c = minimax_ttt(grid, turn, turn, 1 - turn, -math.inf, math.inf)
                time.sleep(delay)
                draw(screen, r, c)
            if player[turn] == 4:
                pygame.display.flip()
                [r, c] = q_learning_ttt(grid, turn)
                time.sleep(delay)
                draw(screen, r, c)
        pygame.display.flip()


def run_connect_four(screen, player, delay, games):
    grid = [[-2 for _ in range(7)] for _ in range(5)]
    grid.append([-1 for _ in range(7)])
    turn = 0
    screenY = screen.get_size()[1]
    Y = screenY // 7
    screenX = screen.get_size()[0]
    end = False
    game = 1
    outcomes = [0, 0, 0]

    def run_clear(screen):
        nonlocal grid
        nonlocal turn
        nonlocal end
        turn = 0
        draw_connect_four(screen, Y)
        grid = [[-2 for _ in range(7)] for _ in range(5)]
        grid.append([-1 for _ in range(7)])
        end = False

    def draw(screen, i):
        nonlocal grid
        nonlocal turn
        for j in range(5, -1, -1):
            if grid[j][i] == -1:
                grid[j][i] = turn
                if j > 0:
                    grid[j - 1][i] = -1
                draw_turn_c4(screen, Y, i, j + 1, turn)
                if check_win() == 'Continue':
                    turn = 1 - turn
                break

    def check_win():
        nonlocal grid
        nonlocal turn
        for i in range(6):
            for j in range(4):
                if grid[i][j] == grid[i][j + 1] == grid[i][j + 2] == grid[i][j + 3] and grid[i][j] != -1 and grid[i][
                    j] != -2:
                    return run_win(grid[i][j], [i, j], [i, j + 1], [i, j + 2], [i, j + 3])
        for j in range(7):
            for i in range(3):
                if grid[i][j] == grid[i + 1][j] == grid[i + 2][j] == grid[i + 3][j] and grid[i][j] != -1 and grid[i][
                    j] != -2:
                    return run_win(grid[i][j], [i, j], [i + 1, j], [i + 2, j], [i + 3, j])
        for i in range(3):
            for j in range(4):
                if grid[i][j] == grid[i + 1][j + 1] == grid[i + 2][j + 2] == grid[i + 3][j + 3] and grid[i][j] != -1 and \
                        grid[i][j] != -2:
                    return run_win(grid[i][j], [i, j], [i + 1, j + 1], [i + 2, j + 2], [i + 3, j + 3])
                if grid[i][j + 3] == grid[i + 1][j + 2] == grid[i + 2][j + 1] == grid[i + 3][j] and grid[i][
                    j + 3] != -1 and grid[i][j + 3] != -2:
                    return run_win(grid[i][j + 3], [i, j + 3], [i + 1, j + 2], [i + 2, j + 1], [i + 3, j])
        if -1 not in np.array(grid).flatten():
            return run_win(2, None, None, None, None)
        else:
            return 'Continue'

    def run_win(player, p1, p2, p3, p4):
        nonlocal turn
        nonlocal grid
        nonlocal end
        nonlocal game
        nonlocal outcomes
        outcomes[player] = outcomes[player] + 1
        game = game + 1
        end = True
        # sX, sY, eX, eY = np.array([sX,sY+1,eX,eY+1])*Y+Y/2
        for i in range(6):
            for j in range(7):
                if grid[i][j] == -1:
                    grid[i][j] = 2
        draw_win_c4(screen, Y, player, p1, p2, p3, p4)
        pygame.display.flip()
        time.sleep(delay)
        pygame.display.flip()
        if game <= games:
            run_clear(screen)
        else:
            print(outcomes)

    buttonSize, labelSize, lineSize = max(3, screen.get_size()[1] // 20 - 6), max(3,
                                                                                  screen.get_size()[1] // 25 - 4), max(
        3, screen.get_size()[1] // 60 - 4)

    draw_connect_four(screen, Y)

    menu = pygame_menu.Menu("", screenX - screenY, screenY, False, position=(100, 0), theme=mytheme)

    grid_buttons = [pygame.Rect(Y * i, 0, Y, Y * 7) for i in range(7)]
    clearButton = menu.add.button('Clear', lambda: run_clear(screen), font_size=labelSize)
    menuButton = menu.add.button('Main Menu', lambda: runMenu(screen, mytheme), font_size=labelSize)
    quizButton = menu.add.button('Quit', pygame_menu.events.EXIT, font_size=labelSize)

    buttons = [clearButton, menuButton, quizButton]
    for button in buttons:
        button.set_margin(max(1, screenY / 60 + 10), max(1, screenY / 40 - 5))
        button.set_border(2, SHADOW, position=('position-south', 'position-east'))
    running = True
    file = 'q0c4.pickle' if player[0] == 3 else 'q0c4.pickle'
    with open(file, 'rb') as f:
        q_table = pickle.load(f)
        while running:
            events = pygame.event.get()
            if menu.is_enabled():
                menu.update(events)
                menu.draw(screen)
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for i in range(7):
                        if grid_buttons[i].collidepoint(event.pos):
                            draw(screen, i)
                            pygame.display.flip()
            if not end:
                if player[turn] == 1:
                    pygame.display.flip()
                    r, c = easy(grid)
                    time.sleep(delay)
                    draw(screen, c)
                if player[turn] ==2:
                    pygame.display.flip()
                    r, c = medium_c4(grid, turn)
                    time.sleep(delay)
                    draw(screen, c)
                if player[turn] == 3:
                    pygame.display.flip()
                    p, r, c = minimax_c4(grid, turn, turn, 1 - turn, -math.inf, math.inf, 5)
                    time.sleep(delay)
                    draw(screen, c)
                if player[turn] == 4:
                    pygame.display.flip()
                    c = q_learning_c4(grid, turn,q_table)
                    time.sleep(delay)
                    draw(screen, c)
            pygame.display.flip()


screen = pygame.display.set_mode((840, 600))
screen.fill(CELL)

mytheme = pygame_menu.themes.THEME_GREEN.copy()
mytheme.background_color = CELL
mytheme.widget_alignment = pygame_menu.locals.ALIGN_LEFT
mytheme.title = False
mytheme.widget_selection_effect.zero_margin()
mytheme.selection_color = TEXT
mytheme.widget_selection_effect = pygame_menu.widgets.HighlightSelection(0)
mytheme.widget_selection_effect.set_background_color(BUTTON)

'''for opp in ['Easy', 'Medium', 'QLearning']:
    for epoch in range(300):
        print('Opponent: '+ opp+ " Training session: " + str(epoch) +' epsilon = '+str(epoch/30) )
        q_learning_train_c4(0, epsilon = epoch/30,opponent=opp)
        q_learning_train_c4(1, epsilon = epoch/30,opponent=opp)'''

'''for epoch in range(100):
        print(" Training session: " + str(epoch) )
        q_learning_train_ttt(0)
        q_learning_train_ttt(1)'''

runMenu(screen, mytheme)

pygame.quit()