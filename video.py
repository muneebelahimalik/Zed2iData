
import pyzed.sl as sl
import cv2
import numpy as np
import os
import time
from playsound import playsound

def save_calibration_parameters(zed, save_directory):
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters

    intrinsic_left = calibration_params.left_cam
    intrinsic_str_left = "Left Camera Intrinsic Parameters:\n" \
                         "fx: {:.2f}, fy: {:.2f}, cx: {:.2f}, cy: {:.2f}\n" \
                         "Distortion Coefficients: {}\n".format(
                             intrinsic_left.fx, intrinsic_left.fy,
                             intrinsic_left.cx, intrinsic_left.cy,
                             intrinsic_left.disto)

    intrinsic_right = calibration_params.right_cam
    intrinsic_str_right = "Right Camera Intrinsic Parameters:\n" \
                          "fx: {:.2f}, fy: {:.2f}, cx: {:.2f}, cy: {:.2f}\n" \
                          "Distortion Coefficients: {}\n".format(
                              intrinsic_right.fx, intrinsic_right.fy,
                              intrinsic_right.cx, intrinsic_right.cy,
                              intrinsic_right.disto)

    translation = calibration_params.stereo_transform.get_translation()
    extrinsic_str = "Extrinsic Parameters (Translation):\n" \
                    "Tx: {:.2f}, Ty: {:.2f}, Tz: {:.2f}\n".format(
                        translation.get()[0], translation.get()[1], translation.get()[2])

    hfov_str = f"Horizontal Field of View (Left Camera): {intrinsic_left.h_fov}\n"

    with open(os.path.join(save_directory, "calibration_params.txt"), 'w') as file:
        file.write(intrinsic_str_left + intrinsic_str_right + extrinsic_str + hfov_str)


def create_save_directory(base_directory):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    existing_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]
    if existing_folders:
        existing_folders.sort()
        last_folder_num = int(existing_folders[-1])
        new_folder_num = last_folder_num + 1
    else:
        new_folder_num = 0

    new_folder_name = f"{new_folder_num:02d}"
    save_directory = os.path.join(base_directory, new_folder_name)
    os.makedirs(save_directory)

    return save_directory

def play_sound():
    playsound("beep.wav")

def main():
    base_directory = "C:/ZED/Maple-SouthEastern/out_test"
    save_directory = create_save_directory(base_directory)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    save_calibration_parameters(zed, save_directory)

    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    # Video writers
    rgb_video_path = os.path.join(save_directory, "RGB_video.avi")
    depth_video_path = os.path.join(save_directory, "Depth_video.avi")
    rgb_writer = None
    depth_writer = None

    # Set up video writers
    frame_rate = 15  # FPS for the video
    width, height = 2208, 1242  # ZED 2i HD2K resolution

    rgb_writer = cv2.VideoWriter(
        rgb_video_path, cv2.VideoWriter_fourcc(*"XVID"), frame_rate, (width, height)
    )
    depth_writer = cv2.VideoWriter(
        depth_video_path, cv2.VideoWriter_fourcc(*"XVID"), frame_rate, (width, height), isColor=False
    )

    print("Recording video. Press Ctrl+C to stop...")

    try:
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                # RGB frame
                rgb_frame = image.get_data()
                rgb_writer.write(rgb_frame)

                # Depth frame
                depth_data = depth.get_data()
                depth_data = np.nan_to_num(depth_data, nan=0, posinf=0, neginf=0)
                depth_image_8bit = cv2.convertScaleAbs(depth_data, alpha=255.0 / depth_data.max())
                depth_writer.write(depth_image_8bit)

                # Play sound during recording
                play_sound()

    except KeyboardInterrupt:
        print("Video recording stopped.")

    # Release resources
    rgb_writer.release()
    depth_writer.release()
    zed.close()

    print(f"Saved RGB and Depth videos to {save_directory}")


if __name__ == "__main__":
    main()