import numpy as np
import pygame as pg
from pygame.locals import *
from Snake import Snake
from Food import Food
from random import randint
import sys
np.set_printoptions(threshold=sys.maxsize)

class Env:

    '''
    player          - 'bot' for simulation and training
                    - 'human' for human play ( render recomanded )
    height & width  - dimensions for the map array
    render          - pygame window representation
    snake           - actual snake object
    food            - food object
    last_distance   - used for reward computing
    map             - np array for RL
    free_spaces     - number of 0 in the map
    '''
    def __init__(self, player='human', height=10, width=10, render=True):
        self.height = height
        self.width = width
        self.map = np.zeros((width, height), dtype=int)
        self.render = render
        if render:
            pg.init()
            self.screen = pg.display.set_mode((self.width, self.height), HWSURFACE|DOUBLEBUF|RESIZABLE)
            pg.display.set_caption('SnakeAI')
        self.running = True
        self.player = player
        self.snake = Snake(3, width // 2, height // 2, width, height)
        self.food = self.place_food()
        self.free_spaces = height * width - self.snake.length   # snake - 3, food 1
        self.last_distance = self.current_distance()

        for p in self.snake.snakeBody:
            self.map[p[0]][p[1]] = 1
        self.map[self.snake.head[0]][self.snake.head[1]] = 2
        self.map[self.food.position[0]][self.food.position[1]] = 3


    @property
    def obs_space(self):
        return (self.height, self.width)

    @property
    def get_obs(self):
        return self.map

    @property
    def action_space(self):
        return (4,)

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    @property
    def action_bound(self):
        return self.action_space[0]

    @property
    def action_sample(self):
        x = np.zeros(self.action_space, dtype = int)
        x[randint(0, self.action_space[0] - 1)] = 1
        return x

    def current_distance(self):
        return np.linalg.norm(np.array(self.snake.head) - np.array(self.food.position))

    def place_food(self):
        x = randint(0, self.height-1)
        y = randint(0, self.width-1)
        food = Food(x, y)
        while food.position in self.snake.snakeBody:
            x = randint (0, self.height - 1)
            y = randint (0, self.width - 1)
            food = Food (x, y)
        return food

    '''
    Render close handling
    '''
    def on_cleanup(self):
        pg.quit()

    '''
    Render cycle handling. It presumes that the surfaces are updated
    '''
    def on_render(self):
        self.screen.fill((0, 0, 0))
        for p in self.snake.snakeBody:
            self.screen.set_at((p[0], p[1]), (255, 255, 255))
        self.screen.set_at(self.food.position, (0, 255, 0))
        self.screen.set_at(self.snake.head, (255, 0, 0))
        pg.display.update()

    '''
    Handles map update, snake movement
    returns observation, step reward and misc. info
    '''
    def on_loop(self):
        info = self.snake.update(self.food.position)
        reward = 0
        distance = self.current_distance()
        if info == 'MOVED':
            if distance <= self.last_distance:
                reward = 1
            else:
                reward = -1
        elif info == 'HIT':
            reward = -10
            self.running = False
        elif info == 'FOOD':
            reward = 10
            self.food = self.place_food()
            self.free_spaces -= 1
        self.last_distance = distance
        self.map = np.zeros((self.width, self.height), dtype=int)
        for p in self.snake.snakeBody:
            self.map[p[0]][p[1]] = 1
        self.map[self.snake.head[0]][self.snake.head[1]] = 2
        self.map[self.food.position[0]][self.food.position[1]] = 3
        return self.map, reward, self.free_spaces == 0 or info == 'HIT', info

    def step(self, action):
        a_dict = {
            0: self.snake.moveLeft,
            1: self.snake.moveUp,
            2: self.snake.moveRight,
            3: self.snake.moveDown
        }
        a_dict[action]()

        m, r, d, i = self.on_loop()
        if self.render:
            self.on_render()
        return m, r, d, i

    '''
    Handles FPS, inputs and update calls
    '''
    def start(self):
        clock = pg.time.Clock()
        if self.player == 'human':
            while self.running:
                print(self.map)
                pg.time.delay(20)
                clock.tick(1)
                pg.event.pump()
                keys = pg.key.get_pressed()
                if (keys[K_RIGHT]):
                    self.snake.moveRight()
                if (keys[K_LEFT]):
                    self.snake.moveLeft()
                if (keys[K_UP]):
                    self.snake.moveUp()
                if (keys[K_DOWN]):
                    self.snake.moveDown()
                if (keys[K_ESCAPE]):
                    self.running = False
                self.on_loop()
                self.on_render()
            self.on_cleanup()
        elif self.player == 'bot':
            print('start function is used only for human play')