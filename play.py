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

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

        state = environment.get_state()

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction = model(
                state_tensor
            )

        action = torch.argmax(
            prediction
        ).item()

        _, _, done = environment.step(
            action
        )

        if done:

            environment.reset()

        # Draw background
        screen.fill(
            (20, 20, 20)
        )

        # Draw snake
        for x, y in environment.snake:

            pygame.draw.rect(
                screen,
                (0, 200, 0),
                (
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

        # Draw food
        x, y = environment.food

        pygame.draw.rect(
            screen,
            (200, 0, 0),
            (
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        pygame.display.flip()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()