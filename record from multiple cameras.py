import cv2
import torch
import threading
import time

from ultralytics import YOLO

# RTSP stream URLs
rtsp_streams = [
    "rtsp://admin:ADMIN2022@81.149.241.73:1024/Streaming/Channels/101",
    "rtsp://admin:ADMIN2022@81.149.241.73:554/Streaming/Channels/101",
    # Add more stream URLs here as needed
]

# Define the output file names for each camera
output_files = [
    "camera_1.avi",
    "camera_2.avi",
    # Add more output file names here as needed
]

frame_saving_reducer_factor = 3
frame_flag = [False for _ in range(len(rtsp_streams))]

yolo_model_path = "Smoke and Fire.pt"
video_saving_fps = 30
min_confidence = 0.7
iou_thresh = 0.8

class_labels = {
    0: "Smoke",
    1: "Fire"
}

print("GPU ENABLED:", torch.cuda.is_available())


# Function to capture video frames from the RTSP stream
def capture_frames(cap, index):

    # Video reader
    cap = cv2.VideoCapture(rtsp_streams[index])

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from camera {index}. Attempting to reconnect...")
            time.sleep(3)
            continue
        frames_buffer[index] = frame
        frame_flag[index] = True


# Function to run object detection on captured frames
def detect_objects(out, index):
    model = YOLO(yolo_model_path)

    while True:

        frame = frames_buffer[index]

        if frame is None:
            time.sleep(0.1)  # Wait for frames to be available
            continue

        whole_win_area = frame.shape[0] * frame.shape[1]
        res = model.predict(frame, conf=min_confidence, iou=iou_thresh)

        # View results
        for r in res[0]:
            area = r.boxes.xywh.tolist()[0][2] * r.boxes.xywh.tolist()[0][3]
            if area < whole_win_area // 10:
                xmin, y_min, x_max, y_max = list(map(int, r.boxes.xyxy.tolist()[0]))
                class_id = r.boxes.cls.tolist()[0]
                name = class_labels[class_id]
                cv2.rectangle(frame, (xmin, y_min - 30), (xmin + 100, y_min), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(frame, name, (xmin, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (xmin, y_min), (x_max, y_max), (0, 0, 255), 2, cv2.LINE_AA)

        if frame_flag[index]:

            # Reduce the size of the frame
            frame = cv2.resize(frame, (frame.shape[1]//frame_saving_reducer_factor, frame.shape[0]//frame_saving_reducer_factor))


            # Write the frame to the output video file
            out.write(frame)

            # Make the flag as False because it is saved
            frame_flag[index] = False

        # Show the frame in a resizable window
        cv2.namedWindow(f'{index}', cv2.WINDOW_NORMAL)

        # Display the window
        cv2.imshow(f'{index}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Initialize frames buffer
frames_buffer = [None] * len(rtsp_streams)

# Create and start threads for capturing frames and running object detection
capture_threads = []
detect_threads = []

for i in range(len(rtsp_streams)):

    # Video reader
    cap = cv2.VideoCapture(rtsp_streams[i])

    # Get the video frame width and height
    frame_width = int(cap.get(3)) // frame_saving_reducer_factor
    frame_height = int(cap.get(4)) // frame_saving_reducer_factor

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # saving_video_name = str("Live Stream" + current_time + "_" + output_files[i])
    saving_video_name = output_files[i]

    out = cv2.VideoWriter(saving_video_name, fourcc, 30.0, (frame_width, frame_height))

    capture_thread = threading.Thread(target=capture_frames, args=(cap, i))
    detect_thread = threading.Thread(target=detect_objects, args=(out, i))

    capture_threads.append(capture_thread)
    detect_threads.append(detect_thread)

    capture_thread.start()
    detect_thread.start()

# Wait for all threads to finish
for capture_thread, detect_thread in zip(capture_threads, detect_threads):
    capture_thread.join()
    detect_thread.join()

print("Video capture and object detection completed for all cameras.")
