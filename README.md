# Motion Detection Tool

[Demo](https://youtu.be/9L3fVsAHmIs)

This Python application is designed to detect motion in a live camera feed or in pre-recorded video files. It utilizes OpenCV for video processing and implements a background subtraction technique to identify moving objects.

## Features

- Motion detection from a live camera feed or a video file.
- Customizable background history and kernel size for erosion operation.
- Option to save the output video with motion detection overlays.

## Installation

Before running the application, ensure Python is installed on your system. The application requires OpenCV, which can be installed via pip.

```bash
pip install opencv-python
```

## Usage

The application can be run with several command-line arguments to specify its behavior:

- Use a live camera feed for motion detection.

```bash
python motion_detection.py --live
```

- Use a video file for motion detection.

```bash
python motion_detection.py --video_path 'path/to/video.mp4'
```

- Customize the history and kernel size for more accurate detection.

```bash
python motion_detection.py --video_path 'path/to/video.mp4' --history 500 --kernel_size 5
```

- Optionally save the output video with motion detection overlays.

```bash
python motion_detection.py --video_path 'path/to/video.mp4' --save_video
```

## Requirements

- Python 3.x
- OpenCV library

## Contributing

Feel free to fork this repository, enhance the motion detection features, or improve the efficiency of the algorithm. Pull requests for improvements and bug fixes are welcome.

## License

MIT
