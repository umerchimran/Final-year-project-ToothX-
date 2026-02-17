import os
import cv2
import numpy as np
import math

# ------------------------- CONFIG -------------------------
INPUT_IMAGE_DIR = r"D:\FYP 2_ANNO\hypodontia\images"
INPUT_LABEL_DIR = r"D:\FYP 2_ANNO\hypodontia\labels"
RESIZE_DIM = 240
CLASS_ID = 0  # hypodontia
ROTATE_STEP = 15  # degrees per rotation

# ------------------------- OUTPUT FOLDERS (ADDED ONLY THIS) -------------------------
OUTPUT_IMAGE_DIR = r"D:\FYP 2_ANNO\hypodontia\prepro_hypo"
OUTPUT_LABEL_DIR = r"D:\FYP 2_ANNO\hypodontia\prepro_labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ------------------------- PREPROCESS -------------------------
def preprocess_image(img):
    img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
    # Skip grayscale images
    if len(img.shape) != 3 or img.shape[2] != 3:
        return None
    # Noise reduction
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Sharpening
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

# ------------------------- GLOBALS -------------------------
img = None
img_copy = None
drawing = False
dragging = False
start_point = None
selected_box_idx = None
dragging_existing = False
dragging_new = False
rotating_box_idx = None

new_boxes = []
existing_boxes = []

# ------------------------- BOX UTILITIES -------------------------
def rect_to_coords(box):
    cx, cy, w, h, angle = box
    angle_rad = np.deg2rad(angle)
    dx = w/2
    dy = h/2
    corners = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated = np.dot(corners,R.T) + np.array([cx,cy])
    return rotated.astype(int)

def draw_boxes_on_image():
    global img_copy
    img_copy = img.copy()
    for box in existing_boxes:
        pts = rect_to_coords(box)
        cv2.polylines(img_copy, [pts], isClosed=True, color=(0,0,255), thickness=2)
        cv2.putText(img_copy, "hypodontia", (pts[0][0], max(pts[0][1]-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    for box in new_boxes:
        pts = rect_to_coords(box)
        cv2.polylines(img_copy, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(img_copy, "hypodontia", (pts[0][0], max(pts[0][1]-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow("Annotate", img_copy)

def point_inside_box(x, y, box):
    pts = rect_to_coords(box)
    pts = pts.reshape((-1,1,2))
    return cv2.pointPolygonTest(pts, (x,y), False) >= 0

# ------------------------- MOUSE CALLBACK -------------------------
def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, dragging, selected_box_idx
    global dragging_existing, dragging_new, new_boxes, existing_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, box in enumerate(new_boxes):
            if point_inside_box(x,y,box):
                dragging_new = True
                selected_box_idx = idx
                start_point = (x,y)
                return
        for idx, box in enumerate(existing_boxes):
            if point_inside_box(x,y,box):
                dragging_existing = True
                selected_box_idx = idx
                start_point = (x,y)
                return
        drawing = True
        start_point = (x,y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            cx = (start_point[0]+x)/2
            cy = (start_point[1]+y)/2
            w = abs(x - start_point[0])
            h = abs(y - start_point[1])
            box = (cx,cy,w,h,0)
            pts = rect_to_coords(box)
            cv2.polylines(temp_img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.imshow("Annotate", temp_img)
        elif dragging_new and selected_box_idx is not None:
            dx = x - start_point[0]
            dy = y - start_point[1]
            cx, cy, w, h, angle = new_boxes[selected_box_idx]
            new_boxes[selected_box_idx] = (cx+dx, cy+dy, w, h, angle)
            start_point = (x,y)
            draw_boxes_on_image()
        elif dragging_existing and selected_box_idx is not None:
            dx = x - start_point[0]
            dy = y - start_point[1]
            cx, cy, w, h, angle = existing_boxes[selected_box_idx]
            existing_boxes[selected_box_idx] = (cx+dx, cy+dy, w, h, angle)
            start_point = (x,y)
            draw_boxes_on_image()

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            end_point = (x, y)
            cx = (start_point[0]+end_point[0])/2
            cy = (start_point[1]+end_point[1])/2
            w = abs(end_point[0]-start_point[0])
            h = abs(end_point[1]-start_point[1])
            new_boxes.append((cx,cy,w,h,0))
            drawing = False
            draw_boxes_on_image()
        elif dragging_new or dragging_existing:
            dragging_new = False
            dragging_existing = False
            selected_box_idx = None

# ------------------------- MAIN LOOP -------------------------
image_files = os.listdir(INPUT_IMAGE_DIR)
idx = 0

while 0 <= idx < len(image_files):
    img_name = image_files[idx]
    img_path = os.path.join(INPUT_IMAGE_DIR, img_name)
    label_path = os.path.join(INPUT_LABEL_DIR, os.path.splitext(img_name)[0]+".txt")

    img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img_raw)
    if img is None:
        print(f"Skipping grayscale or invalid image: {img_name}")
        idx += 1
        continue

    new_boxes = []
    existing_boxes = []

    if os.path.exists(label_path):
        with open(label_path,"r") as f:
            lines = f.readlines()
        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            cx = xc*RESIZE_DIM
            cy = yc*RESIZE_DIM
            w = bw*RESIZE_DIM
            h = bh*RESIZE_DIM
            existing_boxes.append((cx,cy,w,h,0))

    draw_boxes_on_image()
    cv2.setMouseCallback("Annotate", mouse_callback)
    print(f"Annotate {img_name}: 'n' next, 'b' previous, 'u' undo last new box, 'r' rotate selected box, ESC exit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed. Exiting...")
            exit()
        elif key == ord('n'):
            idx += 1
            break
        elif key == ord('b'):
            idx -= 1
            if idx < 0: idx = 0
            break
        elif key == ord('u'):
            if new_boxes:
                new_boxes.pop()
                draw_boxes_on_image()
        elif key == ord('r'):
            if selected_box_idx is not None:
                if dragging_new:
                    cx,cy,w,h,angle = new_boxes[selected_box_idx]
                    new_boxes[selected_box_idx] = (cx,cy,w,h,(angle+ROTATE_STEP)%360)
                elif dragging_existing:
                    cx,cy,w,h,angle = existing_boxes[selected_box_idx]
                    existing_boxes[selected_box_idx] = (cx,cy,w,h,(angle+ROTATE_STEP)%360)
                draw_boxes_on_image()

    # Original saving (unchanged)
    with open(label_path,"w") as f:
        for box in existing_boxes + new_boxes:
            cx,cy,w,h,angle = box
            f.write(f"{CLASS_ID} {cx/RESIZE_DIM} {cy/RESIZE_DIM} {w/RESIZE_DIM} {h/RESIZE_DIM}\n")

    # --------- ADDITIONAL SAVE TO NEW FOLDERS (ONLY ADDED PART) ---------
    output_img_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
    cv2.imwrite(output_img_path, img)

    output_label_path = os.path.join(
        OUTPUT_LABEL_DIR,
        os.path.splitext(img_name)[0] + ".txt"
    )

    with open(output_label_path,"w") as f:
        for box in existing_boxes + new_boxes:
            cx,cy,w,h,angle = box
            f.write(f"{CLASS_ID} {cx/RESIZE_DIM} {cy/RESIZE_DIM} {w/RESIZE_DIM} {h/RESIZE_DIM}\n")

cv2.destroyAllWindows()
