import cv2
import numpy as np
from ultralytics import YOLO
import pickle

# Load YOLOv5 model
model = YOLO('yolov5s.pt')

# Load calibration data (optional)
try:
    with open('calibration_data.pickle', 'rb') as f:
        data = pickle.load(f)
        mtx, dist = data['mtx'], data['dist']
        def undistort(img): return cv2.undistort(img, mtx, dist, None, mtx)
except FileNotFoundError:
    print("[INFO] Calibration data not found. Skipping undistortion.")
    def undistort(img): return img

# Perspective transform for lane detection
def perspective_transform(img):
    h, w = img.shape[:2]
    src = np.float32([[w*0.43, h*0.65], [w*0.58, h*0.65], [w*0.95, h], [w*0.05, h]])
    dst = np.float32([[w*0.25, 0], [w*0.75, 0], [w*0.75, h], [w*0.25, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, Minv

def threshold_lane(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    binary = np.zeros_like(scaled)
    binary[(scaled >= 20) & (scaled <= 100)] = 255
    return binary

def draw_lane(img, binary_warped, Minv):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]
    margin, minpix, nwindows = 100, 50, 9
    leftx_current, rightx_current = leftx_base, rightx_base
    window_height = binary_warped.shape[0] // nwindows
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(rightx) == 0:
        return img

    left_fit, right_fit = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

def estimate_distance(bbox_height):
    if bbox_height == 0:
        return 9999
    return 4000 / bbox_height  # Tuned for demo

# Initialize
speed = 60  # km/h
speed_limit = 70
cap = cv2.VideoCapture("car.mp4")
ret, frame = cap.read()
height, width = frame.shape[:2]
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting.")
        break

    frame = undistort(frame)
    warped, Minv = perspective_transform(frame)
    thresholded = threshold_lane(warped)
    lane_img = draw_lane(frame, thresholded, Minv)

    results = model(frame, verbose=False)[0]
    annotated_frame = lane_img.copy()
    warning_text = ""

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_height = y2 - y1

        # Skip detections in bottom 40% (likely car bonnet)
        if y1 > height * 0.6:
            continue

        distance = estimate_distance(bbox_height)
        label = f'{model.names[cls_id]} {conf:.2f} | {int(distance)} cm'
        color = (0, 255, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if 5 < distance < 20:
            warning_text = "Slow down! Object too close"
        elif 0 < distance < 5:
            warning_text = "STOP! Object too close"
        elif speed > speed_limit:
            warning_text = "SLOW DOWN! Over speed"

    # Overlay speed and warnings
    cv2.putText(annotated_frame, f"Speed: {speed} km/h", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if warning_text:
        cv2.putText(annotated_frame, warning_text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show and save
    cv2.imshow("Autonomous Vehicle", annotated_frame)
    out.write(annotated_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):
        speed += 5
    elif key == ord('-'):
        speed = max(0, speed - 5)

cap.release()
out.release()
cv2.destroyAllWindows()
