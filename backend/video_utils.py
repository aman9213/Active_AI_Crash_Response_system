import cv2

def extract_frames(path, n=8):

    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames