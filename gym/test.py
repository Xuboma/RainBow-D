import pygame
import sys
import time

# Initialize Pygame
pygame.init()

# Global timing variables
end_times = [None, None, None, None]
paused = False
start_times = [time.time(), time.time(), time.time(), time.time()]
pause_times = [0, 0, 0, 0]  # Pause duration

# Window and track constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
TRACK_LENGTH = 800
TRACK_WIDTH = 320
EDGE_MARGIN = 100
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
DOT_RADIUS = 8
BUTTON_POS = (WINDOW_WIDTH // 2 - BUTTON_WIDTH // 2, WINDOW_HEIGHT - 100)  # Centered at the bottom

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (180, 50, 42)
PURPLE = (140, 0, 140)
GRAY = (200, 200, 200)

# Create window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('800m Track Simulation')

# Speeds for each dot
RED_SPEEDS = [5, 7, 9]
BLUE_SPEEDS = [2, 4, 6]
PURPLE_SPEEDS = [5, 6, 7]
GREEN_SPEEDS = [3, 5, 7]


# Function to draw a button
def draw_button(window, text, position, button_size):
    font = pygame.font.Font(None, 30)
    text_render = font.render(text, True, BLACK)
    text_rect = text_render.get_rect(center=(position[0] + button_size[0] // 2, position[1] + button_size[1] // 2))
    button_rect = pygame.Rect(position[0], position[1], button_size[0], button_size[1])
    pygame.draw.rect(window, GRAY, button_rect)
    window.blit(text_render, text_rect)
    return button_rect


# Function to choose speed segment
def choose_speed_segment(current_position, track_length):
    if current_position >= EDGE_MARGIN + track_length * 7 / 9:
        return 2
    elif current_position >= EDGE_MARGIN + track_length * 1 / 9:
        return 1
    else:
        return 0


# Function for race simulation
def race_simulation(window, clock, RED_SPEEDS, BLUE_SPEEDS, PURPLE_SPEEDS, GREEN_SPEEDS):
    global end_times, paused, start_times, pause_times
    speeds = [RED_SPEEDS, BLUE_SPEEDS, PURPLE_SPEEDS, GREEN_SPEEDS]
    dot_x = (WINDOW_WIDTH - TRACK_LENGTH) // 2
    dot_positions_y = [WINDOW_HEIGHT // 2 - TRACK_WIDTH // 4 + i * (TRACK_WIDTH // 4) for i in range(-1, 3)]
    dots_pos = [(dot_x, y + TRACK_WIDTH // 8) for y in dot_positions_y]
    current_positions = [dot_x] * 4
    segment_lengths = [TRACK_LENGTH * 1 / 9, TRACK_LENGTH * 6 / 9, TRACK_LENGTH * 2 / 9]  # Length of each segment

    # Draw pause button
    pause_button = draw_button(window, "Pause", BUTTON_POS, (BUTTON_WIDTH, BUTTON_HEIGHT))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if pause_button.collidepoint(mouse_pos):
                    paused = not paused
                    if paused:
                        for i in range(4):
                            if end_times[i] is None:
                                pause_times[i] = time.time()
                    else:
                        for i in range(4):
                            if end_times[i] is None:
                                start_times[i] += time.time() - pause_times[i]
                                pause_times[i] = 0

        window.fill(WHITE)

        # Draw track and lanes
        pygame.draw.rect(window, BLACK, (EDGE_MARGIN, WINDOW_HEIGHT // 2 - TRACK_WIDTH // 2, TRACK_LENGTH, TRACK_WIDTH),
                         1)
        for i in range(1, 4):
            dash_start_y = WINDOW_HEIGHT // 2 - TRACK_WIDTH // 2 + i * (TRACK_WIDTH // 4)
            for x in range(EDGE_MARGIN, EDGE_MARGIN + TRACK_LENGTH, 20 * 2):
                pygame.draw.line(window, BLACK, (x, dash_start_y), (x + 20, dash_start_y), 2)

        # Draw dots
        for i, color in enumerate([RED, BLUE, PURPLE, GREEN]):
            if end_times[i] is None and not paused:
                segment_index = choose_speed_segment(current_positions[i], TRACK_LENGTH)
                if segment_index == 0:
                    current_time = time.time() - start_times[i]
                    current_positions[i] = dot_x + speeds[i][segment_index] * current_time
                elif segment_index == 1:
                    current_time = time.time() - start_times[i] - segment_lengths[0] / speeds[i][0]
                    current_positions[i] = dot_x + segment_lengths[0] + speeds[i][segment_index] * current_time
                elif segment_index == 2:
                    current_time = time.time() - start_times[i] - (
                                segment_lengths[0] / speeds[i][0] + segment_lengths[1] / speeds[i][1])
                    current_positions[i] = dot_x + segment_lengths[0] + segment_lengths[1] + speeds[i][
                        segment_index] * current_time

                if current_positions[i] >= EDGE_MARGIN + TRACK_LENGTH:
                    end_times[i] = time.time() - start_times[i]
                    current_positions[i] = EDGE_MARGIN + TRACK_LENGTH
            pygame.draw.circle(window, color, (int(current_positions[i]), dots_pos[i][1]), DOT_RADIUS)

        # Draw pause/continue button
        button_text = "Continue" if paused else "Pause"
        pause_button = draw_button(window, button_text, BUTTON_POS, (BUTTON_WIDTH, BUTTON_HEIGHT))

        # Display speed, distance, time, and stage information
        font = pygame.font.Font(None, 35)
        for i, (color, name) in enumerate(zip([RED, BLUE, PURPLE, GREEN], ['Rainbow-D', 'Rainbow', 'DDQN', 'DQN'])):
            speed_segment = choose_speed_segment(current_positions[i], TRACK_LENGTH)
            speed_text1 = font.render(f'{name}:', True, color)
            window.blit(speed_text1, (20+80, 20 + i * 30))

            # Draw solid line before text
            line_length = 60
            solid_line = pygame.Surface((line_length*1.2, 4))
            solid_line.fill(color)
            window.blit(solid_line, (20 - line_length - 10 + 65, 20 + i * 30 + 10))

            speed_text2 = font.render(f'Speed: {speeds[i][speed_segment]}', True, color)
            window.blit(speed_text2, (160 + 80+10, 20 + i * 30))

            distance = min(current_positions[i] - dot_x, TRACK_LENGTH)
            distance_text = font.render(f'Distance: {distance:.2f}m', True, color)
            window.blit(distance_text, (280 + 80+10+10, 20 + i * 30))

            if end_times[i] is not None:
                elapsed_time = end_times[i]
            else:
                elapsed_time = pause_times[i] - start_times[i] if paused else time.time() - start_times[i]
            time_text3 = font.render(f'Time: {elapsed_time:.2f}s', True, color)
            window.blit(time_text3, (510 + 80+10+10+5, 20 + i * 30))

            if speed_segment == 0:
                stage_text1 = font.render(f'Stage: 1', True, color)
            elif speed_segment == 1:
                stage_text1 = font.render(f'Stage: 2', True, color)
            else:
                stage_text1 = font.render(f'Stage: 3', True, color)
            window.blit(stage_text1, (685+80+10+10+5, 20 + i * 30))

        pygame.display.update()
        clock.tick(30)


# Start race simulation
clock = pygame.time.Clock()
race_simulation(window, clock, RED_SPEEDS, BLUE_SPEEDS, PURPLE_SPEEDS, GREEN_SPEEDS)
