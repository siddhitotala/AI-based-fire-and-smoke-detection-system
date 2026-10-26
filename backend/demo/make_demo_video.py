from pathlib import Path
import urllib.request
from PIL import Image, ImageSequence
import cv2
import numpy as np

base = Path(__file__).resolve().parent
base.mkdir(parents=True, exist_ok=True)
gif_path = base / 'fire_demo_1.gif'
mp4_path = base / 'sample.mp4'
url = 'https://raw.githubusercontent.com/luminous0219/fire-and-smoke-detection-yolov8/main/fire%20demo%201.gif'

urllib.request.urlretrieve(url, gif_path)

image = Image.open(gif_path)
frames = [np.array(frame.convert('RGB')) for frame in ImageSequence.Iterator(image)]
if not frames:
    raise RuntimeError('No frames found in demo gif')

height, width = frames[0].shape[:2]
writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
for frame in frames:
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()
print(f'Created {mp4_path}')
