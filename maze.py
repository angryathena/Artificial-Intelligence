import time
import sys
import colorsys
import random

import numpy as np
from mazelib import Maze
from mazelib.generate.CellularAutomaton import CellularAutomaton
import pygame
import pygame_menu
from copy import copy
from time import sleep

CELL = (255, 191, 223)
WALL = (255, 255, 255)
SHADOW = (99, 0, 49)
TEXT = (207, 0, 88)
START = (162, 191, 67)
END = (250, 74, 47)
BUTTON = (255, 219, 237)

sys.setrecursionlimit(2000)
pygame.init()
pygame.display.set_caption('Assignment')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def drawMaze(maze, start, end, screen, cellSize):
    maze[start[0]][start[1]] = 0
    maze[end[0]][end[1]] = 0
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            cell = pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize)
            if maze[row][col] == 1:
                pygame.draw.rect(screen, WALL, cell)
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            cell = pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize)
            if maze[row][col] == 0:
                shadow_offset = 2
                shadow_rect = cell.inflate(shadow_offset * 2, shadow_offset * 2)
                shadow_rect.move_ip(-shadow_offset, -shadow_offset)
                pygame.draw.rect(screen, SHADOW, shadow_rect)
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            cell = pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize)
            if maze[row][col] == 0:
                pygame.draw.rect(screen, CELL, cell)

    (row, col) = start
    pygame.draw.rect(screen, START, pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize))

    (row, col) = end
    pygame.draw.rect(screen, END, (col * cellSize, row * cellSize, cellSize, cellSize))
    maze[start[0]][start[1]] = 1
    maze[end[0]][end[1]] = 1


def runDfs(g, start, end, screen, n, cellSize, delay):
    print(); print()
    tic = time.time()
    path, cells = dfs(g, start, end, [], [])
    toc = time.time()
    print('Depth First Search')
    print()
    print("Time: ", str((toc - tic) * 1000 // 1), ' ms')
    print('Path length: ', str(len(path)), ' nodes')
    print('Nodes visited: ', str(len(cells)))
    path.insert(0, start)
    cells.insert(0, start)
    drawSearch(g, start, end, screen, n, path, cells, cellSize, delay)


def runBfs(g, start, end, screen, n, cellSize, delay):
    print(); print()
    tic = time.time()
    path, cells = bfs(g, start, end, [])
    toc = time.time()
    print('Breadth First Search')
    print()
    print("Time: ", str((toc - tic) * 100000 // 1 / 100), ' ms')
    print('Path length: ', str(len(path)), ' nodes')
    print('Nodes visited: ', str(len(cells)))
    drawSearch(g, start, end, screen, n, path, cells, cellSize, delay)


def runAstar(g, start, end, screen, n, cellSize, delay):
    print(); print()
    tic = time.time()
    path, cells = Astar(g, start, end, [])
    toc = time.time()
    print('A* search')
    print()
    print("Time: ", str((toc - tic) * 1000 // 1), ' ms')
    print('Path length: ', str(len(path)), ' nodes')
    print('Nodes visited: ', str(len(cells)))
    path.insert(0, start)
    drawSearch(g, start, end, screen, n, path, cells, cellSize, delay)


def runVI(g, start, end, screen, n, noise, gamma, cellSize):
    print(); print()
    print('Value Iteration Markov Decision Process')
    print()
    drawMaze(g, start, end, screen, cellSize)
    pygame.display.flip()
    tic = time.time()
    mazeValue, mazePolicy = valueMDP(g, end, noise, gamma)
    toc = time.time()
    drawMDP(g, start, end, screen, n, cellSize, mazeValue, mazePolicy)
    pygame.display.flip()
    path = mdpPath(g, start, end, mazePolicy, 0)
    cells = mdpPath(g, start, end, mazePolicy, noise)
    print("Time: ", str((toc - tic) * 1000 // 1), ' ms')
    print('Path length: ', str(path), ' nodes')
    print('Nodes visited: ', str(cells))


def runPI(g, start, end, screen, n, noise, gamma, cellSize):
    print(); print()
    print('Policy Iteration Markov Decision Process')
    print()
    drawMaze(g, start, end, screen, cellSize)
    pygame.display.flip()
    tic = time.time()
    mazeValue, mazePolicy = policyMDP(g, start, end, noise, gamma)
    toc = time.time()
    drawMDP(g, start, end, screen, n, cellSize, mazeValue, mazePolicy)
    pygame.display.flip()
    path = mdpPath(g, start, end, mazePolicy, 0)
    cells = mdpPath(g, start, end, mazePolicy, noise)
    print("Time: ", str((toc - tic) * 1000 // 1), ' ms')
    print('Path length: ', str(path), ' nodes')
    print('Nodes visited: ', str(cells))


def mdpPath(g, start, end, mazePolicy, noise):
    length = []
    iter = 100 if noise != 0.0 else 1
    for i in range(iter):
        path = [start]
        if start[0] == 0:
            current = (start[0] + 1, start[1])
        elif start[1] == 0:
            current = (start[0], start[1] + 1)
        elif start[0] == len(g) - 1:
            current = (start[0] - 1, start[1])
        else:
            current = (start[0], start[1] - 1)

        while current != end:
            r = random.randint(0, 100)
            path.append(current)
            up = (current[0] - 1, current[1]) if g[current[0] - 1][current[1]] == 0 and (
                current[0] - 1, current[1]) != start else current
            left = (current[0], current[1] - 1) if g[current[0]][current[1] - 1] == 0 and (
                current[0], current[1] - 1) != start else current
            down = (current[0] + 1, current[1]) if g[current[0] + 1][current[1]] == 0 and (
                current[0] + 1, current[1]) != start else current
            right = (current[0], current[1] + 1) if g[current[0]][current[1] + 1] == 0 and (
                current[0], current[1] + 1) != start else current
            neighbours = {'^': [up, left, right], '>': [right, up, down], 'v': [down, left, right],
                          '<': [left, up, down]}
            if r / 100 >= noise:
                current = neighbours[mazePolicy[current[0]][current[1]]][0]
            elif r / 100 >= noise / 2:
                current = neighbours[mazePolicy[current[0]][current[1]]][1]
            else:
                current = neighbours[mazePolicy[current[0]][current[1]]][2]
        length.append(len(path))
    return sum(length) / iter


def drawSearch(g, start, end, screen, n, path, cells, cellSize, delay):
    drawPath(g, start, end, screen, n, cells, cellSize, SHADOW, delay / 2, True)
    drawPath(g, start, end, screen, n, path, cellSize, SHADOW, delay)


def drawPath(g, start, end, screen, n, cells, cellSize, colour2, delay, e=False):
    drawMaze(g, start, end, screen, cellSize)
    prev = cells[0]
    (h, s, v) = (0, 0.42, 0.93)
    for cell in cells:
        h = h + 0.5 / n ** 2
        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        colour = (int(r * 255), int(g * 255), int(b * 255)) if e else colour2
        if pygame.event.peek(pygame.QUIT):
            running = False
            break
        if prev != None:
            (row, col) = prev
            pygame.draw.rect(screen, colour, (col * cellSize, row * cellSize, cellSize, cellSize))
        (row, col) = cell
        if end:
            pygame.draw.rect(screen, START, (col * cellSize, row * cellSize, cellSize, cellSize))
        else:
            pygame.draw.rect(screen, START, (col * cellSize, row * cellSize, cellSize, cellSize))
        prev = cell
        pygame.display.flip()
        time.sleep(delay)
    if e:
        pygame.draw.rect(screen, colour, (col * cellSize, row * cellSize, cellSize, cellSize))
    else:
        pygame.draw.rect(screen, START, (col * cellSize, row * cellSize, cellSize, cellSize))


def drawMDP(maze, start, end, screen, n, cellSize, mazeValue, mazePolicy):
    font = pygame.font.Font(None, 8 + cellSize)
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            if not maze[row][col] == 1:
                cell = pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize)
                (h, s, v) = (0.93, mazeValue[row][col] * 0.9 + 0.1, 1)
                (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
                colour = (int(r * 255), int(g * 255), int(b * 255))
                pygame.draw.rect(screen, colour, cell)
                policy_text = font.render(mazePolicy[row][col], True, SHADOW)
                screen.blit(policy_text, (col * cellSize, row * cellSize))
    (row, col) = start
    # pygame.draw.rect(screen, START, pygame.Rect(col * cellSize, row * cellSize, cellSize, cellSize))

    (row, col) = end
    pygame.draw.rect(screen, END, (col * cellSize, row * cellSize, cellSize, cellSize))


def dfs(maze, current, end, list, visited):
    path = []
    (row, col) = current
    neighbours = [(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]
    for neighbour in neighbours:
        if neighbour == end:
            list.append(neighbour)
            return list, visited
        (row, col) = neighbour
        if row >= len(maze) or col >= len(maze[0]):
            continue
        if maze[row][col] == 0 and neighbour not in visited:
            visited.append(neighbour)
            list.append(neighbour)
            path.extend(dfs(maze, neighbour, end, list, visited)[0])
            if path != []:
                break
    list.pop()
    return path, visited


def bfs(maze, start, end, visited):
    queue = [start]
    parent = {start: None}
    current = queue[0]
    while current != end:
        current = queue[0]
        visited.append(current)
        (row, col) = current
        neighbours = [(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]
        queue.remove(current)
        for neighbour in neighbours:
            (row, col) = neighbour
            if neighbour == end:
                parent[neighbour] = current
                current = neighbour
                break
            if row >= len(maze) or col >= len(maze[0]):
                continue
            if maze[row][col] == 0 and neighbour not in queue and neighbour not in visited:
                parent[neighbour] = current
                queue.append(neighbour)
    path = []
    while current != start:
        path.insert(0, current)
        current = parent[current]
    path.insert(0, start)
    return path, visited


def Astar(maze, start, end, visited):
    gCost = {start: 0}
    hCost = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    cost = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    parent = {start: None}
    current = start
    visited = []
    while current != end:
        current = next(iter(cost))
        visited.append(current)
        (row, col) = current
        cost.pop(current)
        neighbours = [(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]
        for neighbour in neighbours:
            (row, col) = neighbour
            if neighbour == end:
                parent[neighbour] = current
                current = neighbour
                break
            if row >= len(maze) or col >= len(maze[0]):
                continue
            if maze[row][col] == 0 and neighbour not in visited:
                parent[neighbour] = current
                gCost[neighbour] = gCost[current] + 1
                hCost[neighbour] = abs(row - end[0]) + abs(col - end[1])
                cost[neighbour] = gCost[neighbour] + hCost[neighbour]
                cost = dict(sorted(cost.items(), key=lambda item: item[1]))
    path = []
    while current != start:
        path.insert(0, current)
        current = parent[current]
    return path, visited


def valueMDP(grid, end, noise, gamma):
    converged = False
    grid[end[0]][end[1]] = 0
    mazePrev = np.array([[0.0] * len(grid[0])] * len(grid))
    mazePrev[end[0]][end[1]] = 1
    mazeValue = copy(mazePrev)
    mazePolicy = np.array([['^'] * len(grid[0])] * len(grid))
    k = 0
    value = {}
    while not converged:
        k = k + 1
        mazePrev = copy(mazeValue)
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if (row, col) == end or grid[row][col] == 1:
                    continue
                else:
                    neighbours = [['^', row - 1, col], ['<', row, col - 1], ['v', row + 1, col], ['>', row, col + 1],
                                  ['^', row - 1, col], ['>', row, col + 1]]
                    for i in range(4):
                        value[neighbours[i][0]] = 0
                        for j, p in enumerate([noise / 2, 1 - noise, noise / 2]):
                            if neighbours[i + j - 1][1] >= len(grid) or neighbours[i + j - 1][2] >= len(grid[0]):
                                continue
                            if grid[neighbours[i + j - 1][1]][neighbours[i + j - 1][2]] == 0:
                                value[neighbours[i][0]] = value[neighbours[i][0]] + p * gamma * \
                                                          mazePrev[neighbours[i + j - 1][1]][neighbours[i + j - 1][2]]
                            else:
                                value[neighbours[i][0]] = value[neighbours[i][0]] + p * gamma * mazePrev[row][col]
                    value = dict(sorted(value.items(), key=lambda item: item[1]))
                    mazeValue[row][col] = list(value.values())[-1]
                    mazePolicy[row][col] = list(value)[-1]
        if np.max(np.abs(mazePrev - mazeValue)) < 10 ** (-12):
            converged = True
    print('Finished in ', k, ' iterations')
    return mazeValue, mazePolicy


def policyMDP(grid, start, end, noise, gamma):
    converged = False
    grid[end[0]][end[1]] = 0
    mazeValue = np.array([[0.0] * len(grid[0])] * len(grid))
    mazeValue[end[0]][end[1]] = 1
    mazePolicy = np.array([['^'] * len(grid[0])] * len(grid))
    l = 0
    while not converged:
        k = 0
        l = l + 1
        while not converged:
            k = k + 1
            mazePrev = copy(mazeValue)
            for row in range(len(grid)):
                for col in range(len(grid[row])):
                    currentValue = mazePrev[row][col]
                    if (row, col) == end or grid[row][col] == 1:
                        continue
                    up = mazePrev[row - 1][col] if grid[row - 1][col] == 0 and (
                        row - 1, col) != start else currentValue
                    left = mazePrev[row][col - 1] if grid[row][col - 1] == 0 and (
                        row, col - 1) != start else currentValue
                    down = mazePrev[row + 1][col] if grid[row + 1][col] == 0 and (
                        row + 1, col) != start else currentValue
                    right = mazePrev[row][col + 1] if grid[row][col + 1] == 0 and (
                        row, col + 1) != start else currentValue
                    neighbours = {'^': [up, left, right], '>': [right, up, down], 'v': [down, left, right],
                                  '<': [left, up, down]}
                    mazeValue[row][col] = gamma * ((1 - noise) * neighbours[mazePolicy[row][col]][0] + noise / 2 * (
                                neighbours[mazePolicy[row][col]][1] + neighbours[mazePolicy[row][col]][2]))
            if np.max(np.abs(mazePrev - mazeValue)) < 10 ** (-12) or k > 100:
                converged = True
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if (row, col) == end or grid[row][col] == 1:
                    continue
                else:
                    neighbours = [['^', row - 1, col], ['<', row, col - 1], ['v', row + 1, col], ['>', row, col + 1],
                                  ['^', row - 1, col], ['>', row, col + 1]]
                    value = {}
                    for i in range(4):
                        value[neighbours[i][0]] = 0
                        for j, p in enumerate([noise / 2, 1 - noise, noise / 2]):
                            if neighbours[i + j - 1][1] >= len(grid) or neighbours[i + j - 1][2] >= len(grid[0]):
                                continue
                            if grid[neighbours[i + j - 1][1]][neighbours[i + j - 1][2]] == 0:
                                value[neighbours[i][0]] = value[neighbours[i][0]] + p * gamma * \
                                                          mazeValue[neighbours[i + j - 1][1]][neighbours[i + j - 1][2]]
                            else:
                                value[neighbours[i][0]] = value[neighbours[i][0]] + p * gamma * mazeValue[row][col]
                    value = dict(sorted(value.items(), key=lambda item: item[1]))
                    if (list(value.values())[-1] > mazeValue[row][col] and not mazePolicy[row][col] == list(value)[-1]):
                        mazePolicy[row][col] = list(value)[-1]
                        converged = False
    print('Finished in ', l, ' iterations')
    return mazeValue, mazePolicy


def runMenu(screen, mytheme):
    n = '10'
    noise = '0.2'
    gamma = '0.9'

    buttonSize, labelSize, lineSize = max(1, screen.get_size()[1] // 20 - 6), max(1,
                                                                                  screen.get_size()[1] // 25 - 4), max(
        1, screen.get_size()[1] // 60 - 4)

    def setN(size):
        nonlocal n
        n = str(size)

    def setNoise(value):
        nonlocal noise
        noise = str(value)

    def setGamma(value):
        nonlocal gamma
        gamma = str(value)

    def setY(size):
        if size.isdigit() and int(size) >= 300:
            screenY = int(size)
            screenX = 1.4 * screenY
            screen = pygame.display.set_mode((screenX, screenY))
            screen.fill(WALL)
            runMenu(screen, mytheme)

    run = lambda: print("Maze size must be an integer larger than 2.") if not n.isdigit() else print(
        "Gamma must be a number between 0 and 1.") if not isfloat(gamma) else print(
        "Noise must be a number between 0 and 1.") if not isfloat(noise) else print(
        "Maze size must be an integer larger than 2.") if int(n) < 3 else print(
        "Gamma must be a number between 0 and 1.") if float(gamma) < 0 or float(gamma) > 1 else print(
        "Noise must be a number between 0 and 1.") if float(noise) < 0 or float(noise) > 1 else runMaze(screen, int(n),
                                                                                                        float(noise),
                                                                                                        float(gamma))

    main_menu = pygame_menu.Menu('Main Menu', screen.get_size()[0], screen.get_size()[1], theme=mytheme)
    main_menu.add.label('Settings:', font_size=labelSize)
    main_menu.add.text_input('Maze size n: ', default='10', font_size=buttonSize, onchange=setN).set_border(2, SHADOW,
                                                                                                            position=(
                                                                                                                'position-south',
                                                                                                                'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.text_input('MDP Noise: ', default='0.2', font_size=buttonSize, onchange=setNoise).set_border(2,
                                                                                                               SHADOW,
                                                                                                               position=(
                                                                                                                   'position-south',
                                                                                                                   'position-east')).set_margin(
        max(3, screen.get_size()[0] / 60 + 10), max(3, screen.get_size()[0] / 40 - 5)).get_value()
    main_menu.add.text_input('MDP Gamma: ', default='0.9', font_size=buttonSize, onchange=setGamma).set_border(2,
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
    main_menu.add.button('Play', run, font_size=buttonSize).set_border(2, SHADOW, position=(
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


def runMaze(screen, n, noise, gamma):
    m = Maze(42)
    m.generator = CellularAutomaton(n, n, complexity=0.1, density=0.5)
    m.generate()
    m.generate_entrances(True)
    g = m.grid
    start = m.start
    end = m.end
    delay = 5 / n ** 2
    screenY = screen.get_size()[1]
    screenX = screen.get_size()[0]
    cellSize = screenY // (2 * n + 1)
    buttonSize, labelSize, lineSize = max(3, screen.get_size()[1] // 20 - 6), max(3,
                                                                                  screen.get_size()[1] // 25 - 4), max(
        3, screen.get_size()[1] // 60 - 4)

    drawMaze(g, start, end, screen, cellSize)

    menu = pygame_menu.Menu("", screenX - screenY, screenY, False, position=(100, 0), theme=mytheme)
    menu.add.label('', 'l0', font_size=2 * lineSize)
    menu.add.label('Uninformed Search', 'u', font_size=labelSize)
    dfsButton = menu.add.button('DFS', lambda: runDfs(g, start, end, screen, n, cellSize, delay), font_size=buttonSize)
    bfsButton = menu.add.button('BFS', lambda: runBfs(g, start, end, screen, n, cellSize, delay), font_size=buttonSize)
    menu.add.label('', 'l1', font_size=lineSize)
    menu.add.label('Informed Search', 'i', font_size=labelSize)
    astarButton = menu.add.button('A*', lambda: runAstar(g, start, end, screen, n, cellSize, delay),
                                  font_size=buttonSize)
    menu.add.label('', 'l2', font_size=lineSize)
    menu.add.label('MDP Search', 'mdp', font_size=labelSize)
    viButton = menu.add.button('VI', lambda: runVI(g, start, end, screen, n, noise, gamma, cellSize),
                               font_size=buttonSize)
    piButton = menu.add.button('PI', lambda: runPI(g, start, end, screen, n, noise, gamma, cellSize),
                               font_size=buttonSize)
    menu.add.label('', 'l3', font_size=2 * lineSize)

    clearButton = menu.add.button('Clear', lambda: drawMaze(g, start, end, screen, cellSize), font_size=labelSize)
    menuButton = menu.add.button('Main Menu', lambda: runMenu(screen, mytheme), font_size=labelSize)
    quizButton = menu.add.button('Quit', pygame_menu.events.EXIT, font_size=labelSize)

    buttons = [dfsButton, bfsButton, astarButton, viButton, piButton, clearButton, menuButton, quizButton]
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
        pygame.display.flip()


screen = pygame.display.set_mode((840, 600))
screen.fill(WALL)

mytheme = pygame_menu.themes.THEME_GREEN.copy()
mytheme.background_color = CELL
mytheme.widget_alignment = pygame_menu.locals.ALIGN_LEFT
mytheme.title = False
mytheme.widget_selection_effect.zero_margin()
mytheme.selection_color = TEXT
mytheme.widget_selection_effect = pygame_menu.widgets.HighlightSelection(0)
mytheme.widget_selection_effect.set_background_color(BUTTON)

runMenu(screen, mytheme)

pygame.quit()
