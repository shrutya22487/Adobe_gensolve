
pip install cv2

123.0 in [123]

import numpy as np
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs
def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for plotting

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
plot(read_csv('/isolated.csv'))

import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs
def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for plotting

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
def polylines2svg(paths_XYs, svg_path):
 W, H = 0, 0
 for path_XYs in paths_XYs:
  for XY in path_XYs:
    W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
 padding = 0.1
 W, H = int(W + padding * W), int(H + padding * H)
 # Create a new SVG drawing
 dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
 group = dwg.g()

 for i, path in enumerate(paths_XYs):
  colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for plotting

  path_data = []
  c = colours[i % len(colours)]
  for XY in path:
    path_data.append(("M", (XY[0, 0], XY[0, 1])))
    for j in range(1, len(XY)):
      path_data.append(("L", (XY[j, 0], XY[j, 1])))
    if not np.allclose(XY[0], XY[-1]):
      path_data.append(("Z", None))
  group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

 dwg.add(group)
 dwg.save()
 png_path = svg_path.replace('.svg', '.png')
 fact = max(1, 1024 // min(H, W))
 cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact*W, output_height=fact*H, background_color='white')
 return

polylines2svg(read_csv('/isolated.csv'), 'abcd.svg')

import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs
def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for plotting

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
def polylines2svg(paths_XYs, svg_path):
 W, H = 0, 0
 for path_XYs in paths_XYs:
  for XY in path_XYs:
    W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
 padding = 0.1
 W, H = int(W + padding * W), int(H + padding * H)
 # Create a new SVG drawing
 dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
 group = dwg.g()

 for i, path in enumerate(paths_XYs):
  colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Valid SVG color names

  path_data = []
  c = colours[i % len(colours)]
  for XY in path:
    path_data.append(("M", (XY[0, 0], XY[0, 1])))
    for j in range(1, len(XY)):
      path_data.append(("L", (XY[j, 0], XY[j, 1])))
    if not np.allclose(XY[0], XY[-1]):
      path_data.append(("Z", None))
  group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

 dwg.add(group)
 dwg.save()
 png_path = svg_path.replace('.svg', '.png')
 fact = max(1, 1024 // min(H, W))
 cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact*W, output_height=fact*H, background_color='white')
 return

polylines2svg(read_csv('/isolated.csv'), '/abcd.svg')

import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from sklearn.linear_model import LinearRegression
from collections import defaultdict

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for plotting

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
def is_straight_line(XY, threshold=1.0):
    X = XY[:, 0].reshape(-1, 1)
    y = XY[:, 1]
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return np.all(np.abs(residuals) < threshold)

def regularize_straight_line(XY):
    return np.array([XY[0], XY[-1]])

def is_circle(XY, threshold=0.05):
    center = np.mean(XY, axis=0)
    distances = np.linalg.norm(XY - center, axis=1)
    return np.all(np.abs(distances - np.mean(distances)) < threshold)

def regularize_circle(XY):
    center = np.mean(XY, axis=0)
    radius = np.mean(np.linalg.norm(XY - center, axis=1))
    t = np.linspace(0, 2 * np.pi, 100)
    return np.column_stack((center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))

def is_rectangle(XY, threshold=0.1):
    angles = []
    for i in range(len(XY)):
        p1, p2, p3 = XY[i - 1], XY[i], XY[(i + 1) % len(XY)]
        v1, v2 = p1 - p2, p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(np.degrees(angle))
    return len(XY) == 4 and np.allclose(angles, 90, atol=threshold)

def regularize_rectangle(XY):
    center = np.mean(XY, axis=0)
    diffs = XY - center
    abs_diffs = np.abs(diffs)
    half_width = np.max(abs_diffs[:, 0])
    half_height = np.max(abs_diffs[:, 1])
    corners = np.array([[half_width, half_height], [-half_width, half_height], [-half_width, -half_height], [half_width, -half_height]])
    return center + corners

def classify_and_regularize(paths_XYs):
    regularized_paths = defaultdict(list)
    for path in paths_XYs:
        for XY in path:
            if is_straight_line(XY):
                regularized_paths['straight_lines'].append(regularize_straight_line(XY))
            elif is_circle(XY):
                regularized_paths['circles'].append(regularize_circle(XY))
            elif is_rectangle(XY):
                regularized_paths['rectangles'].append(regularize_rectangle(XY))
            else:
                regularized_paths['others'].append(XY)
    return regularized_paths

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Valid SVG color names

    regularized_paths = classify_and_regularize(paths_XYs)
    for i, (shape_type, paths) in enumerate(regularized_paths.items()):
        for path in paths:
            path_data = []
            c = colours[i % len(colours)]
            path_data.append(("M", (path[0, 0], path[0, 1])))
            for j in range(1, len(path)):
                path_data.append(("L", (path[j, 0], path[j, 1])))
            group.add(dwg.path(d=path_data, fill='none', stroke=c, stroke_width=2))

    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')

# Example usage:
# Ensure the path is correct
polylines2svg(read_csv('/isolated.csv'), '/abcd.svg')

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import svgwrite
import cairosvg

# Step 1: Preprocessing the Image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    return edges

# Step 2: Contour Detection
def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Shape Detection Functions
def is_straight_line(contour, threshold=5.0):
    X = contour[:, 0, 0].reshape(-1, 1)
    y = contour[:, 0, 1]
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return np.max(np.abs(residuals)) < threshold

def regularize_straight_line(contour):
    return np.array([contour[0, 0], contour[-1, 0]])

def is_circle(contour, threshold=0.05):
    center, radius = cv2.minEnclosingCircle(contour)
    distances = np.linalg.norm(contour[:, 0, :] - center, axis=1)
    return np.all(np.abs(distances - radius) / radius < threshold)

def regularize_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    center = tuple(map(int, center))
    radius = int(radius)
    return center, radius

def is_rectangle(contour, threshold=0.1):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return cv2.contourArea(box) / cv2.contourArea(contour) > 1 - threshold

def regularize_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

# Step 3: Shape Fitting and Regularization
def regularize_shapes(contours):
    regularized_shapes = []
    for contour in contours:
        if is_straight_line(contour):
            regularized_shapes.append(('line', regularize_straight_line(contour)))
        elif is_circle(contour):
            regularized_shapes.append(('circle', regularize_circle(contour)))
        elif is_rectangle(contour):
            regularized_shapes.append(('rectangle', regularize_rectangle(contour)))
        else:
            regularized_shapes.append(('other', contour))
    return regularized_shapes

# Save to SVG
def shapes_to_svg(shapes, svg_path):
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for shape_type, shape in shapes:
        if shape_type == 'line':
            x1, y1 = shape[0]
            x2, y2 = shape[1]
            group.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black'))
        elif shape_type == 'circle':
            center, radius = shape
            group.add(dwg.circle(center=center, r=radius, stroke='black', fill='none'))
        elif shape_type == 'rectangle':
            group.add(dwg.polygon(points=shape, stroke='black', fill='none'))
        else:
            points = shape[:, 0, :]
            group.add(dwg.polygon(points=points, stroke='black', fill='none'))

    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return

edges = preprocess_image('/abcd.png')
contours = detect_contours(edges)
regularized_shapes = regularize_shapes(contours)
shapes_to_svg(regularized_shapes, '/output.svg')

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import svgwrite
import cairosvg

# Step 1: Preprocessing the Image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    return edges

# Step 2: Contour Detection
def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Shape Detection Functions
def is_straight_line(contour, threshold=5.0):
    X = contour[:, 0, 0].reshape(-1, 1)
    y = contour[:, 0, 1]
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return np.max(np.abs(residuals)) < threshold

def regularize_straight_line(contour):
    return np.array([contour[0, 0], contour[-1, 0]])

def is_circle(contour, threshold=0.05):
    center, radius = cv2.minEnclosingCircle(contour)
    distances = np.linalg.norm(contour[:, 0, :] - center, axis=1)
    return np.all(np.abs(distances - radius) / radius < threshold)

def regularize_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    center = tuple(map(int, center))
    radius = int(radius)
    return center, radius

def is_rectangle(contour, threshold=0.1):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return cv2.contourArea(box) / cv2.contourArea(contour) > 1 - threshold

def regularize_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

# Step 3: Shape Fitting and Regularization
def regularize_shapes(contours):
    regularized_shapes = []
    for contour in contours:
        if is_straight_line(contour):
            regularized_shapes.append(('line', regularize_straight_line(contour)))
        elif is_circle(contour):
            regularized_shapes.append(('circle', regularize_circle(contour)))
        elif is_rectangle(contour):
            regularized_shapes.append(('rectangle', regularize_rectangle(contour)))
        else:
            regularized_shapes.append(('other', contour))
    return regularized_shapes

# Save to SVG
def shapes_to_svg(shapes, svg_path):
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for shape_type, shape in shapes:
        if shape_type == 'line':
            x1, y1 = shape[0]
            x2, y2 = shape[1]
            group.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black'))
        elif shape_type == 'circle':
            center, radius = shape
            group.add(dwg.circle(center=center, r=radius, stroke='black', fill='none'))
        elif shape_type == 'rectangle':
            group.add(dwg.polygon(points=shape, stroke='black', fill='none'))
        else:
            points = shape[:, 0, :]
            group.add(dwg.polygon(points=points, stroke='black', fill='none'))

    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return

edges = preprocess_image('/abcd.png')
contours = detect_contours(edges)
regularized_shapes = regularize_shapes(contours)
shapes_to_svg(regularized_shapes, '/output.svg')

import numpy as np
import svgwrite
import cairosvg
from sklearn.linear_model import LinearRegression

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Shape Detection Functions
def is_straight_line(contour, threshold=5.0):
    X = contour[:, 0].reshape(-1, 1)
    y = contour[:, 1]
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return np.max(np.abs(residuals)) < threshold

def regularize_straight_line(contour):
    return np.array([contour[0], contour[-1]])

def is_circle(contour, threshold=0.05):
    center, radius = cv2.minEnclosingCircle(contour)
    distances = np.linalg.norm(contour - center, axis=1)
    return np.all(np.abs(distances - radius) / radius < threshold)

def regularize_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    center = tuple(map(int, center))
    radius = int(radius)
    return center, radius

def is_rectangle(contour, threshold=0.1):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return cv2.contourArea(box) / cv2.contourArea(contour) > 1 - threshold

def regularize_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

# Step 3: Shape Fitting and Regularization
def regularize_shapes(contours):
    regularized_shapes = []
    for contour in contours:
        if is_straight_line(contour):
            regularized_shapes.append(('line', regularize_straight_line(contour)))
        elif is_circle(contour):
            regularized_shapes.append(('circle', regularize_circle(contour)))
        elif is_rectangle(contour):
            regularized_shapes.append(('rectangle', regularize_rectangle(contour)))
        else:
            regularized_shapes.append(('other', contour))
    return regularized_shapes

# Save to SVG
def shapes_to_svg(shapes, svg_path):
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for shape_type, shape in shapes:
        if shape_type == 'line':
            x1, y1 = shape[0]
            x2, y2 = shape[1]
            group.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black'))
        elif shape_type == 'circle':
            center, radius = shape
            group.add(dwg.circle(center=center, r=radius, stroke='black', fill='none'))
        elif shape_type == 'rectangle':
            group.add(dwg.polygon(points=shape, stroke='black', fill='none'))
        else:
            points = shape[:, :]
            group.add(dwg.polygon(points=points, stroke='black', fill='none'))

    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return

# Example Usage
csv_path = '/isolated.csv'
paths_XYs = read_csv(csv_path)
regularized_shapes = []

for path in paths_XYs:
    for XY in path:
        regularized_shapes.extend(regularize_shapes(XY))

shapes_to_svg(regularized_shapes, '/abcd.svg')