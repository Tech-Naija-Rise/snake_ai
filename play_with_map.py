import math

import pygame
import torch

from environment import SnakeEnvironment
from model import SnakeNetwork

from constants import *


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (
            WIDTH * CELL_SIZE,
            HEIGHT * CELL_SIZE
        )
    )

    pygame.display.set_caption(
        "Snake AI"
    )

    clock = pygame.time.Clock()

    environment = SnakeEnvironment(
        WIDTH,
        HEIGHT
    )

    model = SnakeNetwork()

    model.load_state_dict(
        torch.load(
            "snake_model.pth",
            weights_only=True
        )
    )

    model.eval()

    running = True

    # straight
    # action = -1

    # camera
    # What position to hook the snake (forever following it on that exact spot)
    cam_x = (WIDTH*CELL_SIZE)//2
    cam_y = (HEIGHT*CELL_SIZE)//2
    while running:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_UP:
            #         action = 0

            #     if event.key == pygame.K_RIGHT:
            #         action = 1

            #     if event.key == pygame.K_LEFT:
            #         action = 2

        state = environment.get_state()

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            prediction = model(
                state_tensor
            )

        # If user doesn't give input

        action = torch.argmax(
            prediction
        ).item()

        _, _, done = environment.step(
            action
        )
        # action = 0

        # always keep going straight after going left or right

        if done:
            print("Game over")
            environment.reset()

        # Draw background
        screen.fill(
            (20, 20, 20)
        )

        # Draw snake
        real_head_x = environment.snake[0][0]
        real_head_y = environment.snake[0][1]
        pygame.draw.rect(
            screen,
            (0, 200, 0),
            (
                cam_x,
                cam_y,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        # body_part_idx = 0
        for x, y in environment.snake[1:]:
            print(f"body part at: [{x},{y}]")
           
            illu_x = ((x-real_head_x)*CELL_SIZE)+cam_x
            illu_y = ((y-real_head_y)*CELL_SIZE)+cam_y
            print(f"Real body part loc: [{illu_x},{illu_y}]")
            pygame.draw.rect(
                screen,
                (0, 200, 0),
                (
                    illu_x,
                    illu_y,
                    CELL_SIZE,
                    CELL_SIZE
                )
            )
            
        # Draw food
        food_x, food_y = environment.food
        print(f"food at: [{food_x},{food_y}]")
        formula_x = cam_x
        formula_x = ((food_x-real_head_x)*CELL_SIZE)+cam_x
        formula_y = ((food_y-real_head_y)*CELL_SIZE)+cam_y
        print(f"Real food loc: [{formula_x},{formula_y}]")
        pygame.draw.rect(
            screen,
            (200, 0, 0),
            (
                formula_x,
                formula_y,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        pygame.display.flip()

        clock.tick(FPS)
        # action = -1

    pygame.quit()

# I want to make the entire place move instead of the snake.


if __name__ == "__main__":
    main()
