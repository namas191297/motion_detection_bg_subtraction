import cv2
import numpy as np
import argparse
import sys

# Function to parse commandline arguments.
def parse_args():

    parser = argparse.ArgumentParser(description="Python application to detect motion from a live camera or a video.")
    parser.add_argument('--live', action='store_true', help='Use the camera to detect motion in a livestream.')
    parser.add_argument('--video_path', help='Path of the video to be used for motion detection.')
    parser.add_argument('--history', default=3000, help='No. of previous frames that the background subtraction model should remember.')
    parser.add_argument('--kernel_size', default=10, help='Kernel size to be used for erosion operation.')
    parser.add_argument('--save_video', action='store_true', help='Flag to enable saving the output video.')
    return parser.parse_args()

if __name__ == '__main__':

    # Fetch the arguments
    args = parse_args()
    live = args.live
    vid_path = args.video_path
    history = args.history
    kernel_size = (args.kernel_size, args.kernel_size)
    save_video = args.save_video

    # Decide whether inference needs to be performed on a livestream or a video.
    if not live and vid_path is None:
        print('Either provide --live as an argument or specify the --video_path. Please check --help.')
        sys.exit()
    elif live and vid_path is not None:
        print('You cannot provide both --live and --video_path argument together. Please check --help.')
        sys.exit()

    # Create a VideoCapture object to read the video frame by frame.
    if live:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(vid_path)

    # Ensure that the video capture object has successfully opened the video/stream without error.
    if not cap.isOpened():
        print('Error in streaming video, please check --help.')
        sys.exit()

    # Create a video writer if the video needs to be saved
    writer = None
    if save_video:
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'XVID')
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if live:
            vid_save_name = 'output_live_video.mp4'
        else:
            vid_save_name = 'output_video.mp4'
        writer = cv2.VideoWriter(vid_save_name, fourcc=fourcc_mp4, fps=fps, frameSize=((frame_w, frame_h)))

    # Initialize a statistical model that keeps track of the background
    backgroundSubtractor = cv2.createBackgroundSubtractorKNN(history=history)
    
    # Start the loop the read the video.
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print('Video has ended or error reading frame.')
            break

        # # Apply the background subtractor to the current frame and obtain the foreground mask
        fg_mask = backgroundSubtractor.apply(frame)
        
        # # Apply erosion to the foregroumd mask to remove the noisy bits.
        kernel = np.ones(kernel_size, dtype=np.uint8)
        eroded_fg_mask = cv2.erode(fg_mask, kernel=kernel)
        # Find the non-zero pixels in the binary foreground mask
        non_zero_pixels = cv2.findNonZero(eroded_fg_mask)
        # Check if any non zero pixels were found
        if non_zero_pixels is not None:
            # Find the rectangle bounding the non_zero_pixels
            x,y,w,h = cv2.boundingRect(non_zero_pixels)
            # Draw the rectangle on the frame
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
        # Stack the frame and the mask to display.
        eroded_fg_mask = cv2.merge([eroded_fg_mask, eroded_fg_mask, eroded_fg_mask])
        stacked_op = np.hstack([frame, eroded_fg_mask])
        stacked_op = cv2.resize(stacked_op, (1280,720), cv2.INTER_AREA)
        cv2.imshow('frame', stacked_op)
        if save_video:
            writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    if save_video:
        writer.release()