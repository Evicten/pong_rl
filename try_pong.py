import gymnasium
from pynput import keyboard

env = gymnasium.make("ALE/Pong-v5", render_mode = 'human')
observation = env.reset()

# Global variables to track the paddle movement
paddle_up = False
paddle_down = False

def on_press(key):
    global paddle_up, paddle_down

    if key == keyboard.Key.up:
        paddle_up = True
    elif key == keyboard.Key.down:
        paddle_down = True

def on_release(key):
    global paddle_up, paddle_down

    if key == keyboard.Key.up:
        paddle_up = False
    elif key == keyboard.Key.down:
        paddle_down = False

def play():
    global paddle_up, paddle_down

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    done = False
    while not done:
        env.render()

        # Determine the action based on the paddle movement
        if paddle_up:
            action = 2  # Move paddle up
        elif paddle_down:
            action = 3  # Move paddle down
        else:
            action = 0  # Do nothing

        observation, reward, done, truncated, info = env.step(action)

    listener.stop()
    env.close()

play()
