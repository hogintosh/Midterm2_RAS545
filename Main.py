import cv2
import numpy as np
from collections import deque
import copy
import matplotlib.pyplot as plt
import time
from serial.tools import list_ports
import pydobot
from pydobot.dobot import MODE_PTP
import time
from openai import OpenAI
client = OpenAI(
  api_key=""
)

# --- Adjustable parameters ---
safety_distance_px = 40
circle_clear_radius = 40
border_thickness = 1
dobot_height = -48
# --- initialize dobot --- # 
# Find Dobot port automatically (works on macOS)
available_ports = list_ports.comports()
dobot_port = None
for port in available_ports:
    if "usbmodem" in port.device or "usbserial" in port.device:
        dobot_port = port.device
        break
if dobot_port is None:
    raise Exception("No Dobot Magician Lite detected. Check your USB connection.")
print(f"Connecting to Dobot on {dobot_port}")
device = pydobot.Dobot(port=dobot_port)
device.home()  # go to home position
time.sleep(1)
(pose, joint) = device.get_pose()
print("ready to scan for image!")

# --- Maze detection function (keeps full frame) ---
def detect_maze(frame):
    plain_frame = copy.deepcopy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maze_thresh, maze_region, roi = None, None, None

    if len(contours) >= 1:
        largest = max(contours, key=cv2.contourArea)
        second_largest = sorted(contours, key=cv2.contourArea)[-2] if len(contours) > 1 else None
        to_draw = [largest] + ([second_largest] if second_largest is not None else [])

        #cv2.drawContours(frame, to_draw, -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(largest)
        if second_largest is not None:
            x2, y2, w2, h2 = cv2.boundingRect(second_largest)
            min_x, min_y = min(x, x2), min(y, y2)
            x_max, y_max = max(x + w, x2 + w2), max(y + h, y2 + h)
        else:
            min_x, min_y, x_max, y_max = x, y, x + w, y + h 

        ih, iw = frame.shape[:2]
        min_x, min_y = max(0, min_x-10), max(0, min_y-10)
        x_max, y_max = min(iw, x_max+10), min(ih, y_max+10)

        cv2.rectangle(frame, (min_x, min_y), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, "Detected Maze Boundary", (min_x+25, min_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        maze_thresh = closed
        maze_region = (min_x, min_y, x_max - min_x, y_max - min_y)
        roi = plain_frame  # keep full frame

    return frame, maze_thresh, maze_region, roi

# --- Live camera feed to detect maze ---
cap = cv2.VideoCapture(0)
maze_frame = None

print("Adjust maze position. Press SPACE to capture the maze. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    detected_frame, maze_thresh, maze_region, roi = detect_maze(frame)

    cv2.imshow("Live Feed", detected_frame)

    key = cv2.waitKey(1) & 0xFF

    # USER locks the maze position with SPACE
    if key == ord(' '):
        if maze_region is not None:
            maze_frame = frame.copy()
            print("Maze captured!")
            break
        else:
            print("Maze not detected. Adjust and try again.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if maze_frame is None:
    raise ValueError("No maze detected in video feed!")

img = maze_frame.copy()
print("Maze image captured successfully.")
# continue your processing here...

# --- Run maze detection to highlight boundaries ---
highlighted_frame, maze_thresh, maze_region, roi = detect_maze(img.copy())
if maze_region is None:
    raise ValueError("Maze boundary could not be detected.")

min_x, min_y, width, height = maze_region
x_max = min_x + width
y_max = min_y + height

# --- Create a masked image that zeroes everything outside the bounding box ---
img_masked = np.zeros_like(img)
img_masked[min_y:y_max, min_x:x_max] = img[min_y:y_max, min_x:x_max]

# --- Convert to HSV for color detection (only inside bounding box matters because outside is black) ---
hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)

# Define HSV ranges for colors (kept from your values)
lower_green = np.array([40, 80, 18])
upper_green = np.array([90, 255, 150])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 220])

# --- Create masks (correctly combine both red ranges) ---
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_red =  cv2.inRange(hsv, lower_red2, upper_red2)

# --- Find circle centers (coordinates are absolute because mask is same size as original) ---
def find_circle_center(mask, color_name):
    mask_blurred = cv2.GaussianBlur(mask,(5,5),0)
    contours, _ = cv2.findContours(mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        (x_c, y_c), r = cv2.minEnclosingCircle(cnt)
        if r > 20:  # slightly permissive radius threshold; tweak if needed
            centers.append((int(x_c), int(y_c)))
            print(f"{color_name} circle center at: ({int(x_c)}, {int(y_c)}) radius={r:.1f}")
    return centers

green_centers = find_circle_center(mask_green, "Green")
red_centers = find_circle_center(mask_red, "Red")

# --- Ensure circles are inside the detected bounding box ---
def filter_centers_to_region(centers, region):
    min_x, min_y, w, h = region
    max_x = min_x + w
    max_y = min_y + h
    return [(x, y) for (x, y) in centers if min_x <= x < max_x and min_y <= y < max_y]

green_centers = filter_centers_to_region(green_centers, maze_region)
red_centers = filter_centers_to_region(red_centers, maze_region)

if not (red_centers and green_centers):
    raise ValueError("Couldn't detect both start (red) and goal (green) circles *inside* the maze boundary!")

choice = input("Start from red or green? (type 'R' for red or 'G' for Green): ").strip().lower()

if choice == "r":
    start = red_centers[0]
    goal = green_centers[0]
elif choice == "g":
    start = green_centers[0]
    goal = red_centers[0]
else:
    raise ValueError("Invalid choice! Please type 'red' or 'green'.")

# --- Create binary maze mask from the masked color image ---
gray_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray_masked, (5,5), 0)
_, binary = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY_INV)
# binary: walls likely = 255, free = 0 (because of THRESH_BINARY_INV)

# --- Clear circle regions (so they don't act as walls), then draw them as white (free) ---
for (cx, cy) in green_centers + red_centers:
    # Then draw a white circle marking start/goal as free (choose radius smaller than clear radius)
    cv2.circle(binary, (cx, cy), int(circle_clear_radius*1.05), 0, -1)

# --- Dilate walls to create safety margin ---
kernel_size = 2 * (safety_distance_px // 2) + 1
kernel = np.ones((kernel_size, kernel_size), np.uint8)
maze_mask = cv2.dilate(binary, kernel, iterations=1)

# --- Add border to keep path inside maze ---
h_m, w_m = maze_mask.shape
cv2.rectangle(maze_mask, (0, 0), (w_m+125, h_m+25), (255, 255, 255), border_thickness)

# --- Invert mask so white = free, black = wall for BFS (your BFS expects 255==free) ---
maze_free = cv2.bitwise_not(maze_mask)

# --- BFS pathfinding (same as yours) ---
def bfs_path(maze, start, goal):
    h, w = maze.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque([(start, [start])])
    
    # 8-connected movement: includes diagonals
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if maze[ny, nx] == 255 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append(((nx, ny), path + [(nx, ny)]))
    return None

path = bfs_path(maze_free, start, goal)
def extract_waypoints(path, threshold=50):
    if not path or len(path) < 3:
        return path

    waypoints = [path[0]]  # start
    prev_dx, prev_dy = None, None
    cumulative_distance = 0

    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]

        # Euclidean distance for small fluctuations
        dist = np.sqrt(dx**2 + dy**2)
        cumulative_distance += dist

        # direction changed or distance threshold reached
        if (dx, dy) != (prev_dx, prev_dy) and cumulative_distance >= threshold:
            waypoints.append(path[i - 1])
            cumulative_distance = 0  # reset distance

        prev_dx, prev_dy = dx, dy

    waypoints.append(path[-1])  # goal
    return waypoints
# --- Dobot Movements --- #
waypointPixel = extract_waypoints(path)

prompt = f"""
Take this Python variable:
waypointPixel = {waypointPixel}

Now create a new variable called 'waypointsYX' 
calculated as:
((0.188*y)+194.4, (0.189*x)-71.42) for each (x, y) in waypointPixel.

Return ONLY the resulting Python list for waypointsYX.
"""
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Extract text output
output_text = response.choices[0].message.content.strip()

# --- Optional: Evaluate model output safely if it returns a list ---
try:
    waypointsYX = eval(output_text)  # Converts text like "[(..., ...), (..., ...)]" to Python list
    print("waypointsYX =", waypointsYX)
except:
    print("Model output:", output_text)


#waypointsYX = [((0.188*y)+194.4, (0.189*x)-71.42) for (x, y) in waypointPixel]
for i, (x,y) in enumerate(waypointsYX):
    device.move_to(mode=int(MODE_PTP.MOVL_XYZ),x=x,y=y,z=dobot_height)


# --- Visualization: show boundary, start/goal, path ---
overlay = highlighted_frame.copy()
# mark the bounding box visually
cv2.rectangle(overlay, (min_x, min_y), (x_max, y_max), (255, 0, 0), 2)
cv2.putText(overlay, "Maze Boundary", (min_x, max(min_y-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

cv2.circle(overlay, start, 8, (0, 0, 255), -1)
cv2.circle(overlay, goal, 8, (0, 255, 0), -1)

if path:
    for i in range(1, len(path)):
        cv2.line(overlay, path[i-1], path[i], (0, 255, 255), 2)
else:
    print("⚠️ No path found!")
if waypointPixel:
    for (wx, wy) in waypointPixel:
        cv2.circle(overlay, (wx, wy), 5, (255, 0, 0), -1)  # Blue filled circle
        cv2.circle(overlay, (wx, wy), 8, (255, 0, 0), 1) 
output = cv2.addWeighted(highlighted_frame, 0.6, overlay, 0.8, 0)

# --- Show results ---
cv2.imshow("Detected Maze Boundaries", highlighted_frame)
cv2.imshow("Masked Image (outside bounding box black)", img_masked)
cv2.imshow("Binary Maze Mask (walls white)", binary)
cv2.imshow("Maze Free Space (white=free)", maze_free)
#cv2.imshow('img_masked',img_masked)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Maze Path within Detected Boundary")
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
