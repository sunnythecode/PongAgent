import os
import torch
from torch import nn
import pygame
import sys
import random
import numpy as np

# Game scheme
'''
1. Game starts
2. Agent is provided state, outputs [P_up, P_down]
    P_up, P_down is the probability of moving paddle up or down
3. Sample this probability distribution for up or down action, apply action to paddle object
4. Here is your reward function based on a state:
    If you win +10,000
    If you are at the ball's position +0.1
    If you aren't near the ball's position -0.05
    If you miss the ball -10,000
5. Now after executing the sampled action, calculate a reward
6. Save the reward in a list and continue with the game until a point is won/lost
7. Now execute the training process:
    for each step in rewards calculate the accumulated reward till that pt
        Note: you can scale rewards exponentially based on time to emphasize improving the final reward
    For each step t:
        redo forward prop/generate an pdf
        Calculate log probability of picking the action you did
        Set loss = -1 * log(probability of ur action) * discounted return from that action onwards
        Minimizing your loss aims to increase the probability of actions with higher returns
        Use optimizer to step back
'''


agent = nn.Sequential(
    nn.Linear(5, 30),
    nn.ReLU(),
    nn.Linear(30, 5),
    nn.Sigmoid(),
    nn.Linear(5, 3),
    nn.Softmax(dim=1)
)

optim = torch.optim.Adam(agent.parameters(), lr=0.0005)
discount = 0.99

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 20
# Paddle positions
left_paddle = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
right_paddle = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
# Ball position and velocity
ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
ball_speed_x = 5
ball_speed_y = 5
# Paddle speeds
paddle_speed = 6
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Scores
left_score = 0
right_score = 0
font = pygame.font.Font(None, 36)
#
toggle_rect = pygame.Rect(150, 75, 100, 50)  # Position and size of the toggle
trainingMode = False  # Initial state


# Game loop
tot_epochs = 200
epoch = 0
prev_paddleY = left_paddle.y

while epoch < tot_epochs:
    Actions = []
    Rewards = []
    States = []
    done = False
    #epoch += 1

    while not(done):
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Policy handling
        keys = pygame.key.get_pressed()
        right_paddle.y = ball.y

        # Agent
        #Take in state
        state = np.array([left_paddle.y, 
                        ball.x, 
                        ball.y,
                        ball_speed_x, 
                        ball_speed_y]).reshape(1, -1)
        state = torch.tensor(state).to(torch.float)
        States.append(state)

        probabilities = agent(state)
        #print("probs:", probabilities)
        dist = torch.distributions.Categorical(probs=probabilities)        
        action = dist.sample().item() # O is down, 1 is up

        Actions.append(action)
        prev_dist = abs(left_paddle.y - ball.y)

        if action == 1 and left_paddle.top > 0:
            left_paddle.y -= paddle_speed
        if action == 0 and left_paddle.bottom < HEIGHT:
            left_paddle.y += paddle_speed
        

        # Ball movement
        ball.x += ball_speed_x
        ball.y += ball_speed_y

        # Ball collision with top/bottom walls
        if ball.top <= 0 or ball.bottom >= HEIGHT:
            ball_speed_y *= -1
        agent_hit = ball.colliderect(left_paddle)
        # Ball collision with paddles
        if agent_hit or ball.colliderect(right_paddle):
            ball_speed_x *= -1
            #ball_speed_y += random.uniform(-1.1, 1.1)

        reward = 0
        
        # Ball goes out of bounds
        if ball.left <= 0:
            right_score += 1
            ball.center = (WIDTH // 2, HEIGHT // 2)
            ball_speed_x *= -1
            reward -= 500.0
            done = True
        if ball.right >= WIDTH:
            left_score += 1
            ball.center = (WIDTH // 2, HEIGHT // 2)
            ball_speed_x *= -1
            reward += 100.0
            done = True
        if agent_hit:
            reward += 50.0
        if prev_dist > abs(left_paddle.y - ball.y):
            reward += 0.1  # Small reward for correct movement
        else:
            reward -= 0.3  # Small penalty for incorrect movement
        if left_paddle.y == prev_paddleY:
            reward -= 0.1  # Small time penalty
        

        #print(prev_dist, abs(left_paddle.y - ball.y))
        Rewards.append(reward)
        prev_paddleY = left_paddle.y
        
        # Drawing
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, left_paddle)
        pygame.draw.rect(screen, WHITE, right_paddle)
        pygame.draw.ellipse(screen, WHITE, ball)
        pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

        # Display scores
        left_text = font.render(str(left_score), True, WHITE)
        right_text = font.render(str(right_score), True, WHITE)
        screen.blit(left_text, (WIDTH // 4, 20))
        screen.blit(right_text, (WIDTH * 3 // 4, 20))
        

        # Update display
        pygame.display.flip()
        pygame.time.Clock().tick(60)
    # End game

    # Discounted returns
    #print(sum(Rewards))
    DiscountedRets = []
    for t in range(len(Rewards)):
        gt = 0.0
        for k, r in enumerate(Rewards[t:]):
            gt += (discount ** k) * r
        DiscountedRets.append(gt)
    #print(sum(DiscountedRets))

    for State, Action, gt in zip(States, Actions, DiscountedRets):
        probs = agent(State)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(torch.tensor(Action, dtype=torch.int))

        loss = -log_prob * gt

        optim.zero_grad()
        loss.backward()
        optim.step()
        
