import cv2
import dlib
import numpy as np

# Load the facial landmark detection model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the videos
first_video = cv2.VideoCapture('copy.mp4')
second_video = cv2.VideoCapture('paste.mp4')

# Get video properties (frame rate, dimensions)
fps = second_video.get(cv2.CAP_PROP_FPS)
frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Using first video's width
frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Using first video's height

# Create an output video file with properties from the first video
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process frames from both videos
frame_count = 0
while frame_count < fps * 2:  # Process frames for 2 seconds
    ret1, frame1 = first_video.read()
    ret2, frame2 = second_video.read()

    if not ret1 or not ret2:
        break

    # Convert frames to grayscale for facial landmark detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect faces and facial landmarks in both frames
    detector = dlib.get_frontal_face_detector()
    faces1 = detector(gray1)
    faces2 = detector(gray2)

    # If faces are detected in both frames
    if faces1 and faces2:
        shape1 = predictor(gray1, faces1[0])
        shape2 = predictor(gray2, faces2[0])

        # Extract lip landmarks (assuming landmarks for lips are 48-68)
        lip_landmarks1 = shape1.parts()[48:68]
        lip_landmarks2 = shape2.parts()[48:68]

        # Transfer lip motion from second video to the first video's frame
        for i in range(len(lip_landmarks1)):
            lip_landmarks1[i].x = lip_landmarks2[i].x
            lip_landmarks1[i].y = lip_landmarks2[i].y

        # Draw the modified facial landmarks on the first frame
        pts = np.array([(p.x, p.y) for p in lip_landmarks1], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame1, [pts], True, (0, 255, 0), 2)

        out.write(frame1)
    frame_count += 1

# Release video objects and close output
first_video.release()
second_video.release()
out.release()
cv2.destroyAllWindows()
