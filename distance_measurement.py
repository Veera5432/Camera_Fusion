import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Known width of the checkerboard square in centimeters
object_width_checkerboard = 2.0

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('output12.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()

    # Object detection using YOLO
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Lowered confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Object detection visualization
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate object size based on the known width of the checkerboard square
            fx = width  # Assuming the focal length is equal to the width of the frame
            object_size = (w / width) * object_width_checkerboard

            # Calculate object distance using the formula: distance = fx * object_width / apparent_width
            apparent_width = w
            distance = (fx * object_width_checkerboard) / apparent_width

            # Display object size and distance information
            size_text = f"Object Size: {object_size:.2f} cm"
            distance_text = f"Distance: {distance:.2f} cm"
            cv2.putText(frame, size_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, distance_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Object Size and Distance Measurement", frame)

    # Write the frame to the video file
    video_output.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera, video writer, and close all OpenCV windows
cap.release()
video_output.release()
cv2.destroyAllWindows()
