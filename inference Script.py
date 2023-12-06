import cv2
from ultralytics import YOLO

# todo Parameters needed to be updated according to your requirements
input_video_path = "smokee.mp4"
output_video_path = "New.avi"
yolo_model_path = "Smoke and Fire.pt"
video_saving_fps = 30
min_confidence = 0.2

# Create an object to read
# from camera
video = cv2.VideoCapture(input_video_path)
model = YOLO(yolo_model_path)

# We need to check if the camera
# is opened previously or not
if not video.isOpened():
    print("Error reading video file")
    exit(0)

# We need to set resolutions.
# so, convert them from float to integer.
# frame_width = int(video.get(3))
# frame_height = int(video.get(4))


frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

class_labels = {
    0: "Smoke",
    1: "Fire"
}

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(output_video_path,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         video_saving_fps, size)

# Starting the loop
while True:

    # Fetching the frame from the video
    ret, frame = video.read()

    whole_win_area = frame.shape[0] * frame.shape[1]

    if ret:

        # Making the prediction over the video
        res = model.predict(frame, conf=min_confidence)

        # frame = res[0].plot()

        # View results
        for r in res[0]:
            area = r.boxes.xywh.tolist()[0][2] * r.boxes.xywh.tolist()[0][3]
            if area < whole_win_area // 10:
                xmin, y_min, x_max, y_max = list(map(int, r.boxes.xyxy.tolist()[0]))
                class_id = r.boxes.cls.tolist()[0]
                name = class_labels[class_id]
                cv2.rectangle(frame, (xmin, y_min-30), (xmin+100, y_min), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(frame, name, (xmin, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (xmin, y_min), (x_max, y_max), (0, 0, 255), 2, cv2.LINE_AA)

        result.write(frame)

        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)

        # Press q on keyboard
        # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
