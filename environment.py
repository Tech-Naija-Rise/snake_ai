import random
import numpy as np


UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class SnakeEnvironment:

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height

        self.reset()

    def reset(self):

        self.direction = RIGHT

        self.head = [
            self.width // 2,
            self.height // 2
        ]
        self.snake = [
            self.head,
            (
                self.head[0] - 1,
                self.head[1]
            ),
            (
                self.head[0] - 2,
                self.head[1]
            )
        ]

        self.score = 0

        self.place_food()

        return self.get_state()

    def place_food(self):

        while True:

            food = [
                random.randrange(self.width),
                random.randrange(self.height)
            ]

            if food not in self.snake:
                self.food = food
                break

    def collision(self, position=None):

        if position is None:
            position = self.head

        x, y = position

        # Wall
        if (
            x < 0
            or x >= self.width
            or y < 0
            or y >= self.height
        ):
            # Try no collision
            # pass
            return True

        # Unblock walls
        # if x<0:
        #     self.head[0] = self.width
        # if x>self.width:
        #     self.head[0] = 0

        # if y < 0:
        #     self.head[1] = self.height
        # if y >= self.height:
        #     self.head[1] = 0


        # Body
        if position in self.snake[1:]:
            return True

        return False

    def get_state(self):

        x, y = self.head

        direction_index = DIRECTIONS.index(
            self.direction
        )

        left_direction = DIRECTIONS[
            (direction_index - 1) % 4
        ]

        right_direction = DIRECTIONS[
            (direction_index + 1) % 4
        ]

        straight = (
            x + self.direction[0],
            y + self.direction[1]
        )

        left = (
            x + left_direction[0],
            y + left_direction[1]
        )

        right = (
            x + right_direction[0],
            y + right_direction[1]
        )

        state = [

            # Danger
            self.collision(straight),
            self.collision(left),
            self.collision(right),

            # Direction
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,

            # Food
            self.food[0] < x,
            self.food[0] > x,
            self.food[1] < y,
            self.food[1] > y
        ]

        return np.array(
            state,
            dtype=np.float32
        )

    def step(self, action):

        """
        action:
            0 = straight
            1 = right
            2 = left
        """

        direction_index = DIRECTIONS.index(
            self.direction
        )

        if action == 1:

            direction_index = (
                direction_index + 1
            ) % 4

        elif action == 2:

            direction_index = (
                direction_index - 1
            ) % 4

        self.direction = DIRECTIONS[
            direction_index
        ]

        x, y = self.head

        dx, dy = self.direction

        self.head = [
            x + dx,
            y + dy
        ]

        self.snake.insert(
            0,
            self.head
        )

        # Death
        if self.collision():
            return (
                self.get_state(),
                -10,
                True
            )

        # Food
        # print(self.head, self.food)

        if list(self.head) == list(self.food):
            self.score += 1
            print(f"Bingo: {self.score}")

            self.place_food()

            reward = 10

        else:

            self.snake.pop()

            reward = 0

        return (
            self.get_state(),
            reward,
            False
        )