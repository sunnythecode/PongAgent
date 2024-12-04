import pygame
import sys
from nn import neural_net
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle dimensions
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

toggle_rect = pygame.Rect(150, 75, 100, 50)  # Position and size of the toggle
trainingMode = False  # Initial state

#Model:
agent = neural_net([5, 20, 10, 1])
#agent = neural_net([8, 2, 1, 1])


# Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if toggle_rect.collidepoint(event.pos):
                    trainingMode = not trainingMode  # Toggle the boolean

    # Policy handling
    keys = pygame.key.get_pressed()
    right_paddle.y = ball.y

    # Agent
    #Take in state
    state = np.array([left_paddle.y, 
                    ball.x, 
                    ball.y,
                    ball_speed_x, 
                    ball_speed_y]).T.reshape(-1, 1)
    acts = agent.forward_prop(state)
    

    action = acts[-1][0][0][0]
    print("ACTION", action)

    if action >= 0.5 and left_paddle.top > 0:
        left_paddle.y -= paddle_speed
    if action < 0.5 and left_paddle.bottom < HEIGHT:
        left_paddle.y += paddle_speed

    optimal_move = 1.0 if ball.y < left_paddle.y else 0.0
    print(optimal_move)

    if trainingMode:
        print("training...")
        agent.train([state], [np.array([[optimal_move]])], 1, 0.4)
    else:
        print("Pressed")


    
    """if keys[pygame.K_UP] and right_paddle.top > 0:
        right_paddle.y -= paddle_speed
    if keys[pygame.K_DOWN] and right_paddle.bottom < HEIGHT:
        right_paddle.y += paddle_speed"""

    # Ball movement
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with top/bottom walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1

    # Ball collision with paddles
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        ball_speed_x *= -1
        ball_speed_y += random.uniform(-1.1, 1.1)

    # Ball goes out of bounds
    if ball.left <= 0:
        right_score += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ball_speed_x *= -1
    if ball.right >= WIDTH:
        left_score += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ball_speed_x *= -1

    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, left_paddle)
    pygame.draw.rect(screen, WHITE, right_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
    # Render text on the toggle
    if trainingMode:
        pygame.draw.rect(screen, GREEN, toggle_rect)
        text = "In_Train"
    else:
        pygame.draw.rect(screen, RED, toggle_rect)
        text = "In_Test"

    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=toggle_rect.center)
    screen.blit(text_surface, text_rect)

    # Display scores
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (WIDTH // 4, 20))
    screen.blit(right_text, (WIDTH * 3 // 4, 20))
    

    # Update display
    pygame.display.flip()
    pygame.time.Clock().tick(60)
