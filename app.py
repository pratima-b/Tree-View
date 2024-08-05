from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
import os

class ImageSeg:
    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(path)

    def find_red_marks(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def count_red_marks(self):
        contours = self.find_red_marks()
        return len(contours)

    def mark_red_marks(self):
        contours = self.find_red_marks()
        marked_img = np.copy(self.img)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(marked_img, f'Tree {i+1}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return marked_img

    def color_filter(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([30, 40, 40])
        upper_bound = np.array([90, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        return filtered_img

    def preprocess_img(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred_img, 50, 150)
        return edges

    def post_process(self, edge_img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel)
        return opened_img

    def count_trees_without_red_marks(self):
        filtered_img = self.color_filter()
        edge_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(edge_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        return num_labels - 1

    def mark_trees_without_red_marks(self):
        filtered_img = self.color_filter()
        edge_img = self.preprocess_img(filtered_img)
        processed_img = self.post_process(edge_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_img, connectivity=8)
        marked_img = np.copy(self.img)
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(marked_img, f'Tree {i}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return marked_img

    def count_trees(self):
        red_mark_count = self.count_red_marks()
        if red_mark_count > 0:
            return red_mark_count
        else:
            return self.count_trees_without_red_marks()

    def mark_trees(self):
        red_mark_count = self.count_red_marks()
        if red_mark_count > 0:
            return self.mark_red_marks()
        else:
            return self.mark_trees_without_red_marks()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the image using ImageSeg class
    image_seg = ImageSeg(file_path)
    tree_count = image_seg.count_trees()
    marked_img = image_seg.mark_trees()

    # Save processed image to a file
    processed_file_path = os.path.join('processed', file.filename)
    cv2.imwrite(processed_file_path, marked_img)

    # Return JSON response
    return jsonify({
        'processed_image_url': f'/processed/{file.filename}',
        'tree_count': tree_count
    })

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory('processed', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    app.run(debug=True)
