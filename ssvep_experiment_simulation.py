# +
import pygame
import random
import time
import pyautogui

#initialize pygame
pygame.init()

#setting display dimensions
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("OpenVEP-SSVEP Data Collection & FBTRCA Model Application Visualization")

#defining colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

#defining position of squares (cross layout)
square_size = 100
square_positions = {
    0: (WIDTH//2 - square_size//2, HEIGHT//2 - 150),  #top (8 Hz)
    1: (WIDTH//2 - square_size//2, HEIGHT//2 + 50),   #bottom (10 Hz)
    2: (WIDTH//2 - 150, HEIGHT//2 - square_size//2),  #left (12 Hz)
    3: (WIDTH//2 + 50, HEIGHT//2 - square_size//2)    #right (15 Hz)
}

#defining frequencies
frequencies = {0: "8 Hz: Move Up", 1: "10 Hz: Move Down", 2: "12 Hz: Move Left", 3: "15 Hz: Move Right"}

def draw_squares(highlighted=None, phase="SSVEP Data Collection", round_num=1):
    """drawing squares & highlighting active one"""
    screen.fill(WHITE)
    
    for idx, pos in square_positions.items():
        color = GREEN if idx == highlighted else BLACK
        pygame.draw.rect(screen, color, (*pos, square_size, square_size), 5)
    
    #displaying phase text (w/ dynamic centering)
    font = pygame.font.Font(None, 36)
    phase_text = f"{phase} - Round {round_num}"
    text_surface = font.render(phase_text, True, BLACK)

    #calculating x-position dynamically based on text width
    text_rect = text_surface.get_rect(center=(WIDTH // 2, 30))
    screen.blit(text_surface, text_rect)
    
    pygame.display.flip()
    

def data_collection_phase():
    """simulating 3 full rounds of data collection before moving to FBTRCA model application"""
    print("\n### Starting SSVEP Data Collection Phase ###")
    
    for round_num in range(1, 4):  #3 rounds
        print(f"Starting Data Collection Round {round_num}")
        for _ in range(10):  #simulating 10 trials per round
            chosen = random.choice([0, 1, 2, 3])
            draw_squares(highlighted=chosen, phase="SSVEP Data Collection", round_num=round_num)
            time.sleep(0.5)
            draw_squares(phase="SSVEP Data Collection", round_num=round_num)  #clearing highlight
            time.sleep(0.5)
    
    print("### SSVEP Data Collection Completed ###\n")
    time.sleep(3)  #pausing before moving to FBTRCAmodel application

def model_application_phase():
    """simulating 3 full rounds of FBTRCA model application after all SSVEP data collection is done"""
    print("### Starting FBTRCA Model Application Phase ###")

    screen_width, screen_height = pyautogui.size() # Get the size of the primary monitor.
    pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=0.5)
    current_x, current_y = pyautogui.position() # Get the XY position of the mouse.

    for round_num in range(1, 4):  #3 rounds
        print(f"Starting FBTRCA Model Application Round {round_num}")
        

        user_target = random.choice([0, 1, 2, 3])  #user fixating on one
        #flashing simulation (all squares flicker, as in real model use)
        for _ in range(5):
            draw_squares(highlighted=random.choice([0, 1, 2, 3]), phase="FBTRCA Model Application", round_num=round_num)
            time.sleep(1)
        
        #model prediction (correctly selecting the user’s intended square)
        draw_squares(highlighted=user_target, phase=f"FBTRCA Model Prediction: {frequencies[user_target]}", round_num=round_num)
        
        if user_target == 2:
            print("Moving left")
            new_x = max(current_x - 200, 0)
            pyautogui.moveTo(new_x, current_y, duration=1)
        elif user_target == 3:
            print("Moving right")
            screen_width, _ = pyautogui.size()
            new_x = min(current_x + 200, screen_width)
            pyautogui.moveTo(new_x, current_y, duration=1)
        elif user_target == 0:
            print("Moving up")
            new_y = max(current_y - 200, 0)
            pyautogui.moveTo(current_x, new_y, duration=1)
        elif user_target == 1:
            print("Moving down")
            screen_height = pyautogui.size().height
            new_y = min(current_y + 200, screen_height)
            pyautogui.moveTo(current_x, new_y, duration=1)
            
        pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=0.5)

        time.sleep(2)

    print("### FBTRCA Model Application Completed ###")
    time.sleep(2)

def main():
    """running full visualization with complete SSVEP data collection first & FBTRCA model application phases second"""
    screen.fill(WHITE)
    pygame.display.flip()
    
    data_collection_phase()
    model_application_phase()

    print("Simulation complete")
    pygame.quit()

if __name__ == "__main__": 
    main()

