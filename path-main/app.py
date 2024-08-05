from flask import Flask, request, render_template, send_file
import numpy as np
import heapq
from PIL import Image, ImageDraw, ImageOps
import io

app = Flask(__name__)

class OptimalPathing:
    def __init__(self, img):
        self.img = img

    def create_graph(self, binary_image):
        rows, cols = binary_image.shape
        graph = {}
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only 4 neighbors for simplicity

        for i in range(rows):
            for j in range(cols):
                neighbors = []
                for dx, dy in neighbor_offsets:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        weight = 1 + (255 - binary_image[ni, nj]) / 255.0  # Simple weight calculation
                        neighbors.append(((ni, nj), weight))
                graph[(i, j)] = neighbors
        return graph

    def trace_path(self, parents, start, target):
        path = []
        current = target
        while current != start:
            path.append(current)
            current = parents[current]
        path.append(start)
        path.reverse()
        return path

    def ComputeDjikstra(self, start_pixel=(0, 0), target_pixel=None):
        if target_pixel is None:
            target_pixel = (self.img.shape[0] - 1, self.img.shape[1] - 1)
        graph = self.create_graph(self.img)
        parents = {}
        heap = [(0, start_pixel)]
        visited = set()

        while heap:
            cost, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            if current == target_pixel:
                break

            for neighbor, weight in graph.get(current, []):
                if neighbor not in visited:
                    parents[neighbor] = current
                    heapq.heappush(heap, (cost + weight, neighbor))

        shortest_path = self.trace_path(parents, start_pixel, target_pixel)

        # Visualize the image and the shortest path using PIL
        img = Image.fromarray(self.img).convert('RGB')
        draw = ImageDraw.Draw(img)
        if shortest_path:
            for i in range(len(shortest_path) - 1):
                start_point = (shortest_path[i][1], shortest_path[i][0])
                end_point = (shortest_path[i + 1][1], shortest_path[i + 1][0])
                draw.line([start_point, end_point], fill="blue", width=2)

        # Save the result to a BytesIO object
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        image = Image.open(file).convert('L')  # Convert to grayscale
        img_array = np.array(image)

        # Create instance of OptimalPathing
        obj = OptimalPathing(img_array)

        # Compute Dijkstra's path
        result_img = obj.ComputeDjikstra()

        # Return the result image
        return send_file(result_img, mimetype='image/png')

    return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)
