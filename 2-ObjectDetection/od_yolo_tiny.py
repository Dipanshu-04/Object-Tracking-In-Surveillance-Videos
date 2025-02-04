from imageai.Detection import VideoObjectDetection
import os
import time

# Start timing
start_time = time.time()

# Set the execution path
execution_path = os.getcwd()


def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")



video_detector = VideoObjectDetection()
video_detector.setModelTypeAsTinyYOLOv3()
# Path to the model file
model_path = os.path.join(execution_path, "models/tiny-yolov3.pt")
if not os.path.exists(model_path):
    print("Model file not found. Please download tiny-yolov3.pt and place it in the 'models' directory.")
else:
    video_detector.setModelPath(model_path)
    video_detector.loadModel()

# Path to input video
input_video_path = os.path.join(execution_path, "videos/TrafficVid1.mp4")
if not os.path.exists(input_video_path):
    print(f"Input video file not found at {input_video_path}. Please provide a valid video file.")
else:
    # Detect objects in the video
    output_video_path = os.path.join(execution_path, "output_video2.mp4")
    video_detector.detectObjectsFromVideo(
        input_file_path=input_video_path,
        output_file_path=output_video_path,
        frames_per_second=10,
        per_second_function=forSeconds,
        per_frame_function=forFrame,
        per_minute_function=forMinute,
        minimum_percentage_probability=30
    )

    # End timing and calculate the duration
    end_time = time.time()
    execution_duration = end_time - start_time

    print("Video saved at:", output_video_path)
    print("Time taken to run the code:", execution_duration, "seconds")



    