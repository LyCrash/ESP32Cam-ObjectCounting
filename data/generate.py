import requests
import cv2

url = "https://www.pexels.com/download/video/12116094/?fps=29.97&h=720&w=1280"
response = requests.get(url, stream=True)
with open("video.mp4", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)

# Replace with the path to your downloaded video
video_path = "video.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if the video has opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
cut_frame = int(frame_count / 4)  # Frame to cut the video at

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('short_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Read and write frames until the cut point
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_num == cut_frame:
        break
    out.write(frame)
    frame_num += 1

# Release everything
cap.release()
out.release()