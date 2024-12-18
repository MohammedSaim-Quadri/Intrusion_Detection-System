import pyautogui
import time

print("Move your mouse to the desired GUI element to get its coordinates.")
try:
    while True:
        x, y = pyautogui.position()  # Get current mouse position
        print(f"Mouse position: x={x}, y={y}", end="\r")  # Print position
        time.sleep(0.1)  # Update every 100ms
except KeyboardInterrupt:
    print("\nStopped tracking.")
