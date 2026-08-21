import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from environment import SnakeEnvironment
from model import SnakeNetwork
from constants import *


class Agent:

    def __init__(self):

        self.model = SnakeNetwork()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LR
        )

        self.loss_function = nn.MSELoss()

        self.memory = deque(
            maxlen=MAX_MEMORY
        )

        self.games = 0

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def get_action(self, state):

        # Exploration decreases over time
        epsilon = max(5, 120 - self.games)

        if random.randint(0, 200) < epsilon:

            return random.randrange(3)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction = self.model(
                state_tensor
            )

        return torch.argmax(
            prediction
        ).item()

    def train_step(
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

        reward = torch.tensor(
            reward,
            dtype=torch.float32
        )

        prediction = self.model(state)

        target = prediction.clone().detach()

        if done:

            target[action] = reward

        else:

            with torch.no_grad():

                next_prediction = self.model(
                    next_state
                )

            target[action] = (
                reward
                + GAMMA *
                torch.max(next_prediction)
            )

        loss = self.loss_function(
            prediction,
            target
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:

            batch = random.sample(
                self.memory,
                BATCH_SIZE
            )

        else:

            batch = self.memory

        for experience in batch:

            self.train_step(
                *experience
            )


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

    environment = SnakeEnvironment()

    record = 0
    print("Starting")
    while True:

        state = environment.reset()

        done = False

        # let the steps be limited
        steps_limit = 5000
        stepss = 0
        
        while not done:
            action = agent.get_action(
                state
            )
            # print(action)

            next_state, reward, done = (
                environment.step(action)
            )


            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train_step(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state


            if stepss>steps_limit:
                # done = True
                break
            stepss+=1
        try:
            print(f"Finished game in: {stepss} steps")
            agent.games += 1

            agent.train_long_memory()

            score = environment.score

            if score > record:

                record = score

                torch.save(
                    agent.model.state_dict(),
                    "snake_model.pth"
                )

            if agent.games % 10 == 0:

                print(
                    f"Games: {agent.games} "
                    f"| Score: {score} "
                    f"| Record: {record}"
                )

        except KeyboardInterrupt:
            torch.save(
                agent.model.state_dict(),
                "snake_model.pth"
            )


if __name__ == "__main__":

    train()