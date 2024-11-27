import cv2
import numpy as np
import json

def load_output_from_file(output_path):
    with open(output_path, "r") as f:
        data = json.load(f)
    return data

def visualize_output(output_data, save_path="testedimg.png"):
    # Load the original image
    image_path = output_data["raw_file"]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 360))

    # Retrieve lanes from output data
    lanes = output_data["lanes"]

    # Iterate through each lane and draw it on the image
    for lane in lanes:
        # Draw points along the lane (use x, y coordinates if valid)
        for i in range(1, len(lane)):
            x1, y1 = lane[i - 1], i - 1  # lane index serves as y-coordinate
            x2, y2 = lane[i], i
            # Only draw if x and y are within bounds
            if all(0 <= val < 640 for val in [x1, x2]) and all(0 <= val < 360 for val in [y1, y2]):
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save the resulting image with lane markings
    cv2.imwrite(save_path, image)
    print(f"Output saved as {save_path}")

if __name__ == "__main__":
    output_path = "model_output.json"
    
    # Load output from the JSON file
    model_output = load_output_from_file(output_path)
    
    # Visualize and save the output
    visualize_output(model_output, "testedimg.png")