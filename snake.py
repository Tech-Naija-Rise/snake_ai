from model import SnakeNetwork
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# =========================
# SETTINGS
# =========================


from constants import *
WIDTH = WIDTH*CELL_SIZE
HEIGHT = HEIGHT*CELL_SIZE
# =========================
# DIRECTIONS
# =========================

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


# =========================
# NEURAL NETWORK
# =========================


# =========================
# AGENT
# =========================


class Agent:

    def __init__(self):

        self.n_games = 0

        self.epsilon = 0
        self.gamma = GAMMA

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = SnakeNetwork()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LR
        )

        self.criterion = nn.MSELoss()

    def get_action(self, state):

        # Exploration decreases over time
        self.epsilon = max(0, 80 - self.n_games)

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)

        else:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            prediction = self.model(state_tensor)

            move = torch.argmax(prediction).item()

        final_move[move] = 1  # type: ignore

        return final_move

    def remember(self, state, action, reward, next_state, done):

        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        next_state = torch.tensor(
            next_state,
            dtype=torch.float32
        )

        action = torch.tensor(
            action,
            dtype=torch.long
        )

        reward = torch.tensor(
            reward,
            dtype=torch.float32
        )

        prediction = self.model(state)

        target = prediction.clone().detach()

        if done:

            target[action.argmax()] = reward

        else:

            next_prediction = self.model(next_state)

            target[action.argmax()] = (
                reward
                + self.gamma * torch.max(next_prediction)
            )

        loss = self.criterion(
            prediction,
            target
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:

            mini_sample = random.sample(
                self.memory,
                BATCH_SIZE
            )

        else:

            mini_sample = self.memory

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in mini_sample:

            self.train_short_memory(
                state,
                action,
                reward,
                next_state,
                done
            )


# =========================
# SNAKE GAME
# =========================

class SnakeGame:

    def __init__(self):

        pygame.init()

        self.lock_cam_to_head = False

        self.display = pygame.display.set_mode(
            (
                WIDTH,
                HEIGHT
            )
        )

        pygame.display.set_caption(
            "Neural Network Snake"
        )

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):

        self.direction = RIGHT

        self.head = (
            WIDTH // 2,
            HEIGHT // 2
        )

        self.snake = [
            self.head,
            (
                self.head[0] - CELL_SIZE,
                self.head[1]
            ),
            (
                self.head[0] - CELL_SIZE * 2,
                self.head[1]
            )
        ]

        self.score = 0

        self.food = []

        self.place_food()

    def place_food(self):

        x = random.randrange(
            0,
            WIDTH,
            CELL_SIZE
        )

        y = random.randrange(
            0,
            HEIGHT,
            CELL_SIZE
        )

        self.food = [x, y]

        if self.food in self.snake:

            self.place_food()

    def is_collision(self, point=None):

        if point is None:

            point = self.head

        x, y = point

        # Wall collision
        if (
            x < 0
            or x >= WIDTH
            or y < 0
            or y >= HEIGHT
        ):
            return True

        # Body collision
        if point in self.snake[1:]:

            return True

        return False

    def play_step(self, action):

        # Handle window events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    self.lock_cam_to_head = not self.lock_cam_to_head

        # Change direction
        clockwise = DIRECTIONS.index(self.direction)

        if action == [1, 0, 0]:

            new_direction = DIRECTIONS[clockwise]

        elif action == [0, 1, 0]:

            new_direction = DIRECTIONS[
                (clockwise + 1) % 4
            ]

        else:

            new_direction = DIRECTIONS[
                (clockwise - 1) % 4
            ]

        self.direction = new_direction

        # Move
        x, y = self.head

        dx, dy = self.direction

        self.head = (
            x + dx * CELL_SIZE,
            y + dy * CELL_SIZE
        )

        self.snake.insert(
            0,
            self.head
        )

        # Collision
        if self.is_collision():

            return -10, True, self.score

        # Food
        if list(self.head) == list(self.food):

            self.score += 1

            self.place_food()

            reward = 10

        else:

            self.snake.pop()

            reward = 0

        # Draw
        self.update_ui()

        self.clock.tick(FPS)

        return reward, False, self.score

    def update_ui(self):

        # lock at screen center
        position_to_lock = [WIDTH//2, HEIGHT//2]
        self.display.fill((0, 0, 0))
        if not self.lock_cam_to_head:
            # Snake
            for block in self.snake:

                pygame.draw.rect(
                    self.display,
                    (0, 200, 0),
                    pygame.Rect(
                        block[0],
                        block[1],
                        CELL_SIZE,
                        CELL_SIZE
                    )
                )

            # Food
            pygame.draw.rect(
                self.display,
                (200, 0, 0),
                pygame.Rect(
                    self.food[0],
                    self.food[1],
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

        else:
            head = self.snake[0]
            # head itself
            pygame.draw.rect(
                self.display,
                (0, 255, 100),
                pygame.Rect(
                    position_to_lock[0],
                    position_to_lock[1],
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

            for block in self.snake[1:]:
                pygame.draw.rect(
                    self.display,
                    (0, 200, 0),
                    pygame.Rect(
                        ((block[0]-head[0]))+position_to_lock[0],
                        ((block[1]-head[1]))+position_to_lock[1],
                        CELL_SIZE,
                        CELL_SIZE
                    )
                )

            # Food
            pygame.draw.rect(
                self.display,
                (200, 0, 0),
                pygame.Rect(
                    ((self.food[0]-head[0]))+position_to_lock[0],
                    ((self.food[1]-head[1]))+position_to_lock[1],
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

        pygame.display.flip()

# =========================
# STATE
# =========================


def get_state(game):

    head = game.head

    x, y = head

    direction = game.direction

    # Relative directions
    clockwise = DIRECTIONS.index(direction)

    dir_left = DIRECTIONS[
        (clockwise - 1) % 4
    ]

    dir_right = DIRECTIONS[
        (clockwise + 1) % 4
    ]

    # Points directly ahead, left and right
    point_straight = (
        x + direction[0] * CELL_SIZE,
        y + direction[1] * CELL_SIZE
    )

    point_left = (
        x + dir_left[0] * CELL_SIZE,
        y + dir_left[1] * CELL_SIZE
    )

    point_right = (
        x + dir_right[0] * CELL_SIZE,
        y + dir_right[1] * CELL_SIZE
    )

    state = [

        # Danger straight
        game.is_collision(point_straight),

        # Danger left
        game.is_collision(point_left),

        # Danger right
        game.is_collision(point_right),

        # Current direction
        direction == LEFT,
        direction == RIGHT,
        direction == UP,
        direction == DOWN,

        # Food location
        game.food[0] < x,
        game.food[0] > x,
        game.food[1] < y,
        game.food[1] > y
    ]

    return np.array(
        state,
        dtype=int
    )


# =========================
# TRAINING
# =========================

def train():

    agent = Agent()

    try:
        agent.model.load_state_dict(
            torch.load(
                "snake_model.pth",
                weights_only=True
            )
        )
    except Exception as e:
        print(f"{e}")

    game = SnakeGame()

    record = 0
    stepss = 0
    steps_limit = 100_000

    
    while True:
        try:    
            stepss += 1
            state_old = get_state(game)

            final_move = agent.get_action(
                state_old
            )

            reward, done, score = game.play_step(
                final_move
            )

            state_new = get_state(game)

            agent.train_short_memory(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            agent.remember(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )
            if stepss > steps_limit:
                done = True

            
            if done:

                game.reset()

                agent.n_games += 1

                agent.train_long_memory()

                if score >= record:
                    print("Saving model state")

                    record = score

                    torch.save(
                        agent.model.state_dict(),
                        "snake_model2.pth"
                    )

                print(
                    "Game:",
                    agent.n_games,
                    "Score:",
                    score,
                    "Record:",
                    record,
                    "Total Steps:",
                    stepss
                )


        except KeyboardInterrupt as k:
            print("Interrupted, Saving model...")
            torch.save(
                agent.model.state_dict(),
                "snake_model_cutoff.pth"
            )
            break

if __name__ == "__main__":

    train()
