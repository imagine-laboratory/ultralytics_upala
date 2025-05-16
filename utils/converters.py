import numpy as np

def convert_to_mot_format(results, frame_id=0):
    """
    Convert YOLO output to MOT format.

    Parameters:
        boxes: ultralytics.engine.results.Boxes object
        frame_id: Index of the frame (default is 0 for a single image)

    Returns:
        list of strings in MOT format.
    """
    mot_output = []

    # Extract bounding box data from the boxes object
    boxes_data = results.boxes.xyxy.numpy()  # Convert to numpy array
    confidences = results.boxes.conf.numpy()
    class_indices = results.boxes.cls.numpy()

    # Convert each box to MOT format
    for idx in range(len(boxes_data)):
        x1, y1, x2, y2 = boxes_data[idx]
        width = x2 - x1
        height = y2 - y1
        confidence = confidences[idx]
        class_id = int(class_indices[idx])  # Convert class index to integer if necessary

        # Unique ID assignment for this example
        object_id = idx + 1  

        mot_output.append(f"{frame_id}, {object_id}, {x1}, {y1}, {width}, {height}, {confidence:.2f}, {class_id}")

    return mot_output

