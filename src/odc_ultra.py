import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(r"C:\Projects\obj-detection-counting\scene3.mp4")
assert cap.isOpened(), "Error reading video file"

region_points = [(200,50), (200, 550)]                                      # for scene 3
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangular region
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region
# region_points = [ #for scene 1
#     (50, 150),   # left top
#     (400, 150),   # right top
#     (400, 350),  # right bottom
#     (50, 350)    # left bottom
# ]
# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output-3.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model=r"C:\Projects\obj-detection-counting\output_5\kaggle\working\custom_sack_detection\weights\best.pt",
    classes=[0],  # class 0 = "sack" (the only class in custom model)
    tracker="bytetrack.yaml",
    show_in=True,
    show_out=True,
)

# process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
# cv2.destroyAllWindows()  # destroy all opened windows