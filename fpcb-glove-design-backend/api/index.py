from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import io
from PIL import Image
import numpy as np
import cv2
import os
import itertools
import json
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon as shapPolygon, LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import shapely
import svgwrite
from lxml import etree # Import lxml.etree for SVG modification
from svg2mod.importer import Svg2ModImport
from svg2mod.exporter import Svg2ModExportLatest, DEFAULT_DPI
import shlex
import sys
import shutil
import mediapipe as mp # Imported mediapipe for global helper functions
from skimage import measure # Imported measure for global contour_hand
from typing import Dict, List, Union
import zipfile

app = FastAPI()

app.mount("/output", StaticFiles(directory="output"), name="output")

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"], # Note: will need to edit when hosting
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.pixels_per_cm_width = 118.15655396016675 # 1
app.pixels_per_cm_height = 115.81961345740872 # 1
app.trace_start_distance = 1
app.bottom_edge_distance = 2

# --- Global Helper Functions ---

def smooth_hand_mask(binary_mask, blur_kernel_size=5, morph_kernel_size=3):
    """
    Smooths out the hand binary mask by applying Gaussian blur and morphological operations.
    :param binary_mask: 2D binary mask (1s for hand, 0s for background).
    :param blur_kernel_size: Size of the kernel for Gaussian blur.
    :param morph_kernel_size: Size of the kernel for morphological operations.
    :return: Smoothed binary mask.
    """
    # Apply Gaussian blur to smooth out local details
    smoothed_mask = cv2.GaussianBlur(binary_mask, (blur_kernel_size, blur_kernel_size), 0)
    
    # Threshold the image to convert it back to binary after blurring
    _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    # Create a kernel for morphological operations (a square or elliptical kernel works well)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    # Apply morphological opening (erosion followed by dilation) to remove small noise
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

    # Optionally, apply closing (dilation followed by erosion) to fill small gaps
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)

    return smoothed_mask

def contour_hand(binary_image, boundary=-1, num_points=1000):
    """
    Finds the largest contour in a binary image within a specified boundary
    and upsamples its points.
    :param binary_image: 2D binary mask.
    :param boundary: Y-coordinate to cut the image for contour detection.
    :param num_points: Number of points to upsample the contour to.
    :return: Upsampled contour points (numpy array) and line segments (list of tuples).
    """
    # Crop image based on boundary
    binary_image_cropped = binary_image[:boundary, :]
    
    # Normalize and find contours
    if binary_image_cropped.sum() == 0: # Avoid division by zero if image is all black
        return np.array([]), []
    
    normalized_image = binary_image_cropped / 255.0 # Normalize to 0.0 or 1.0
    
    contours = measure.find_contours(normalized_image, level=0.5)
    
    points = []
    line_segments = []
    
    if not contours: # Handle case where no contours are found
        return np.array([]), []

    # Find the largest contour
    largest_contour = max(contours, key=lambda c: len(c))
    contours_to_process = [largest_contour] # Process only the largest one

    for contour in contours_to_process:
        # Handle cases where contour might be too small
        if len(contour) < 2:
            continue

        # Calculate cumulative distances along the contour
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        
        if cumulative_distances[-1] == 0: # Handle zero-length contours
            continue

        # Interpolate to upsample points
        desired_distances = np.linspace(0, cumulative_distances[-1], num_points)
        interpolated_points = np.array([
            np.interp(desired_distances, cumulative_distances, contour[:, dim])
            for dim in range(contour.shape[1])
        ]).T
        
        points.extend(interpolated_points)
        
        # Create line segments for visualization (not directly used for PCB)
        for i in range(len(points) - 1):
            line_segments.append((i, i + 1))
        
        if points: # Ensure points exist before trying to connect last to first
            line_segments.append((len(points)-1, 0)) # Close the loop
    
    return np.array(points), line_segments

def point_along_line(A, B, t, landmarks):
    """
    Finds a point t% along the line segment from A to B using vector projection.

    Args:
    A, B (tuple): Start and end points of the line segment (x, y).
    t (float): Fraction of the way along the segment (0 to 1).

    Returns:
    array: Coordinates of the interpolated point.
    """
    return np.array(landmarks[A]) + t * (np.array(landmarks[B]) -np.array(landmarks[A]))

def point_along_line_distance(A, B, t, landmarks):
    """
    Finds a point t units along the line segment from A to B.

    Args:
    A, B (tuple): Start and end points of the line segment (x, y).
    t (float): Distance along the segment from point A.
    landmarks (dict): Dictionary of points with coordinates as (x, y).

    Returns:
    array: Coordinates of the interpolated point.
    """
    A_coords = np.array(landmarks[A])
    B_coords = np.array(landmarks[B])
    
    # Compute the vector from A to B
    AB_vector = B_coords - A_coords
    
    # Compute the total length of the segment
    segment_length = np.linalg.norm(AB_vector)
    
    # Find the interpolation factor (t_ratio) as t / segment_length
    t_ratio = t / segment_length if segment_length != 0 else 0
    
    # Calculate the point at the given length along the segment
    return A_coords + t_ratio * AB_vector

def compute_perpendicular_vector(vector):
    """
    Compute a perpendicular vector in 2D.
    :param vector: Input vector (x, y).
    :return: Perpendicular vector (rotated 90 degrees counterclockwise).
    """
    return np.array([-vector[1], vector[0]])

def calculate_rectangle(joint1, joint2, width):
    """
    Given two joints as the length of the rectangle and a width,
    calculate the rectangle's four corners.

    :return: 4 points of the rectangle, in clockwise order starting from the top left point
    """
    # Vector from joint1 to joint2
    length_vector = np.array(joint2) - np.array(joint1)
    length = np.linalg.norm(length_vector)
    if length == 0:
        return None  # Avoid division by zero
    
    # Unit vector along the length
    unit_length_vector = length_vector / length
    
    # Perpendicular vector (rotated 90 degrees)
    perp_vector = np.array([-unit_length_vector[1], unit_length_vector[0]]) * width / 2
    
    # Calculate the corners of the rectangle
    p1 = joint1 + perp_vector
    p2 = joint1 - perp_vector
    p3 = joint2 - perp_vector
    p4 = joint2 + perp_vector
    
    return np.array([p1, p2, p3, p4], dtype=np.int32)

# Modified from Jupyter Notebook: Expects pixel_landmarks_list (list of [x,y] pixel coordinates) directly
def get_finger_polygons(pixel_landmarks_list: List[List[float]], image_shape: tuple, width=10):
    """
    Computes polygons for finger segments based on pixel coordinates of landmarks.
    :param pixel_landmarks_list: List of (x,y) pixel coordinates of landmarks.
    :param image_shape: Tuple (height, width) of the image (used for context, not scaling here as landmarks are already pixel coords).
    :param width: Base width for the polygons.
    :return: Dictionary of regions with their joints and calculated rectangles.
    """
    landmarks = pixel_landmarks_list # Landmarks are already pixel coordinates, so no need to scale
    
    joint_pairs = {
        't1': (point_along_line(4,3,0.1,landmarks), point_along_line(4,3,0.9,landmarks)),
        't2': (point_along_line(3,2,0.1,landmarks), point_along_line(3,2,0.9,landmarks)),
    }

    ratios={
        2:0.87, 3:0.89, 4:0.94, 5:0.87
    }

    for finger_idx in range(2,6):
        mcp_idx = 5 + (finger_idx-2) * 4
        pip_idx = mcp_idx + 1
        dip_idx = pip_idx + 1
        tip_idx = dip_idx + 1
        joint_pairs['d'+str(finger_idx)+'p1'] = (point_along_line(tip_idx, dip_idx, 0.05, landmarks),point_along_line(tip_idx, dip_idx, 0.9, landmarks))
        joint_pairs['d'+str(finger_idx)+'p2'] = (point_along_line(dip_idx, pip_idx, 0.1, landmarks),point_along_line(dip_idx, pip_idx, 0.9, landmarks))
        p2Length = np.linalg.norm(np.array(landmarks[pip_idx])-np.array(landmarks[dip_idx]))
        joint_pairs['d'+str(finger_idx)+'p3'] = (point_along_line(pip_idx, mcp_idx, 0.1, landmarks),point_along_line_distance(pip_idx, mcp_idx, p2Length*ratios[finger_idx], landmarks))

    regionDict={}
    for region in joint_pairs.keys():
        joints = joint_pairs[region]
        joint1 = joints[0]
        joint2 = joints[1]
        calculated_width = np.linalg.norm(joint1 - joint2) * 0.5 

        rectangle = calculate_rectangle(joint1, joint2, calculated_width)
        if rectangle is not None:
            regionDict[region] = (joints, rectangle)
    
    return regionDict

def sample_and_connect_polygon_sides(polygon, num_points=3):
    """
    Sample equal spaced points on the two vertical sides of a polygon and connect them with lines.
    
    :param polygon: A polygon defined by four points (point1, point2, point3, point4).
    :param num_points: Number of equal spaced points to sample on each side.
    :return: A list of lines connecting the sampled points.
    """
    # Extract the points defining the polygon
    point1, point2, point3, point4 = polygon
    
    # Interpolate points on side1 (point1 -> point4) and side2 (point2 -> point3)
    side1_points = np.linspace(point1, point4, num_points)
    side1_points = side1_points[1:-1]
    side2_points = np.linspace(point2, point3, num_points)
    side2_points = side2_points[1:-1]
    
    # Connect corresponding points
    connecting_lines = [[side1_points[i], side2_points[i]] for i in range(num_points-2)]
    
    return connecting_lines

def sample_and_connect_polygon_other_sides(polygon, num_points=5):
    """
    Sample equal spaced points on the two vertical sides of a polygon and connect them with lines.
    
    :param polygon: A polygon defined by four points (point1, point2, point3, point4).
    :param num_points: Number of equal spaced points to sample on each side.
    :return: A list of lines connecting the sampled points.
    """
    # Extract the points defining the polygon
    point1, point2, point3, point4 = polygon
    
    # Interpolate points on side1 (point1 -> point4) and side2 (point2 -> point3)
    side1_points = np.linspace(point1, point2, num_points)
    side1_points = side1_points[1:-1]
    side2_points = np.linspace(point4, point3, num_points)
    side2_points = side2_points[1:-1]
    
    # Connect corresponding points
    connecting_lines = [[side1_points[i], side2_points[i]] for i in range(num_points-2)]
    
    return connecting_lines

def draw_electrodes(ax, horizontal_electrodes, vertical_electrodes):
    """
    Draw the horizontal electrode traces (blue) and vertical electrode traces (cyan)
    
    :param ax: Figure axes
    :param horizontal_electrodes: Array of electrodes as (point1, point2)
    :param vertical_electodes: Array of electrodes as (point1, point2)
    """
    for point1, point2 in horizontal_electrodes:
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue', linewidth=1)

    for point1, point2 in vertical_electrodes:
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='cyan', linewidth=1) 

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def project(A, B, C):
    """
    Projects point C onto the infinite line through points A and B.

    Args:
    A, B, C (tuple): Coordinates of points A, B, and C as (x, y).

    Returns:
    array: Coordinates of the projected point.
    """
    A_coords = np.array(A)
    B_coords = np.array(B)
    C_coords = np.array(C)
    
    # Vector from A to B
    AB = B_coords - A_coords
    
    # Vector from A to C
    AC = C_coords - A_coords
    
    # Project AC onto AB
    AB_norm_sq = np.dot(AB, AB)  # Squared norm of AB
    if AB_norm_sq == 0:
        # A and B are the same point, projection is undefined; return A
        return A_coords
    
    projection_factor = np.dot(AC, AB) / AB_norm_sq  # Scalar projection
    projection = A_coords + projection_factor * AB  # Projected point
    
    return projection

def line_orthogonal_to_AB(A, B, t=1):
    """
    Finds a point C on the line AC orthogonal to AB.
    
    Parameters:
        A (np.ndarray): Coordinates of point A as a numpy array [x1, y1].
        B (np.ndarray): Coordinates of point B as a numpy array [x2, y2].
        t (float): A parameter to control the distance of C from A.
    
    Returns:
        np.ndarray: Coordinates of point C as a numpy array [x, y].
    """
    # Direction vector of AB
    d = B - A
    
    # Orthogonal vector to AB (swap and negate one component)
    orthogonal_vector = np.array([-d[1], d[0]])
    
    # Point C is obtained by moving along the orthogonal vector
    C = A + t * orthogonal_vector
    return C

def line_intersection(p1, p2, p3, p4):
    """
    Finds the intersection of two lines: (p1, p2) and (p3, p4).
    
    Parameters:
        p1, p2: Points on the first line (numpy arrays).
        p3, p4: Points on the second line (numpy arrays).
    
    Returns:
        np.ndarray or None: Intersection point [x, y] as a numpy array, or None if lines are parallel.
    """
    # Line vectors
    d1 = p2 - p1
    d2 = p4 - p3
    
    # Form the matrix for the linear system
    A = np.array([d1, -d2]).T
    b = p3 - p1
    
    # Check if lines are parallel
    if np.linalg.det(A) == 0:
        return None  # No intersection
    
    # Solve the linear system for the intersection point
    t = np.linalg.solve(A, b)
    intersection = p1 + t[0] * d1
    return intersection

# Modified from Jupyter Notebook: Expects pixel_landmarks_list (list of [x,y] pixel coordinates) directly
def create_palm_region(pixel_landmarks_list: List[List[float]], regionDict: dict, contour: np.ndarray, boundary: int, image_shape: tuple, pixels_per_cm_width: float) -> np.ndarray:
    """
    Creates a polygon representing the palm region based on pixel coordinates of landmarks and hand contour.
    :param pixel_landmarks_list: List of (x,y) pixel coordinates of landmarks.
    # ... other params ...
    """
    landmarks = pixel_landmarks_list
    
    pinkyPatch3 = regionDict['d5p3'][1]
    indexPatch3 = regionDict['d2p3'][1]
    
    new0 = np.array([contour[-1][0] - 1.5*pixels_per_cm_width, float(boundary)], dtype=np.float32)
    new1 = np.array([contour[0][0] + 1.5*pixels_per_cm_width, float(boundary)], dtype=np.float32)

    new3 = point_along_line(1,2, 0.5,landmarks)
    
    contourTop_point = min(contour, key=lambda p: euclidean_distance(regionDict['d2p1'][1][0], p))
    contourBottom_point = min(contour, key=lambda p: euclidean_distance(regionDict['d2p3'][1][3], p))
    new35 = project(contourTop_point, contourBottom_point, np.array(landmarks[2])) # Use np.array(landmarks[2]) as it's a list.

    new2 = project(indexPatch3[0],indexPatch3[3],np.array(landmarks[5]))
    new2 = new2 + (indexPatch3[3]-new2)*0.5

    ringPatch3 = regionDict['d4p3'][1]

    new6 =  np.array(landmarks[13]) + (np.array(landmarks[9])-np.array(landmarks[13]))*0.5
    new6 = new6 + (ringPatch3[3]-new6)*0.5

    new4 = project(pinkyPatch3[1], pinkyPatch3[2], np.array(landmarks[17]))
    new4 = new4 + (pinkyPatch3[2]-new4)*0.5
    new8 = new0+(new4-new0)*0.5

    palmRegion = np.array([new0, new1, new3, new35, new2, new6, new4, new8], dtype=np.float32)
    return palmRegion

def interpolate_path(points, num_samples, fractions=None):
    """
    Interpolates points along a path defined by `points`.

    Args:
    - points (list of ndarray): Ordered points defining the path (e.g., [p1, p2, p3, ...]).
    - num_samples (int): Total number of points to sample along the path.
    - fractions (list of float, optional): If provided, must be of length `num_samples`.
    Each value should be between 0 and 1, representing the fractional distance
    along the total path to place each sample (for uneven sampling).

    Returns:
    - sampled_points (list of ndarray): Interpolated points along the path.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required to interpolate a path.")
    
    # Compute segment lengths
    segment_lengths = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    total_length = sum(segment_lengths)

    # Compute cumulative segment start positions
    cumulative_lengths = np.cumsum([0] + segment_lengths)

    # Compute distances to sample at
    if fractions is not None:
        if len(fractions) != num_samples:
            raise ValueError("Length of fractions must match num_samples.")
        if not all(0 <= f <= 1 for f in fractions):
            raise ValueError("All elements in fractions must be between 0 and 1.")
        distances = [f * total_length for f in fractions]
    else:
        distances = np.linspace(0, total_length, num_samples)

    # Interpolate sampled points
    sampled_points = []
    for dist in distances:
        for i in range(len(points) - 1):
            if cumulative_lengths[i] <= dist <= cumulative_lengths[i + 1]:
                t = (dist - cumulative_lengths[i]) / segment_lengths[i]
                interpolated_point = (1 - t) * points[i] + t * points[i + 1]
                sampled_points.append(interpolated_point)
                break

    return sampled_points

def sample_traces_palm(palmPolygon, numVertical=13, numHorizontal=9):
    #Extract points
    p1, p2, p3, p4, p5, p6,p7,p8 = palmPolygon
    # Interpolate points on side1 (point1 -> point4) and side2 (point2 -> point3)
    side1_points = interpolate_path(np.array([p1,p8,p7]), numHorizontal, fractions=[0,0.05,0.13,0.21,0.29,0.75,0.85,0.95,1.0])
    side1_points = side1_points[1:-1]
    middle_points = interpolate_path(np.array([p2,p6]), numHorizontal, fractions=[0,0.05,0.13,0.21,0.29,0.75,0.85,0.95,1.0])
    middle_points = middle_points[1:-1]
    side2_points = interpolate_path(np.array([p3,p4,p5]), numHorizontal, fractions=[0,0.05,0.13,0.21,0.29,0.75,0.85,0.95,1.0])
    side2_points = side2_points[1:-1]
    
    # Connect corresponding points
    horizontalTraces = [[side1_points[i], middle_points[i], side2_points[i]] for i in range(numHorizontal-2)]

    bottompoints = interpolate_path(np.array([p1,p2,p3]),numVertical)
    bottompoints = bottompoints[1:-1]
    newmiddle_points = np.linspace(p8,p4, numVertical)
    newmiddle_points = newmiddle_points[1:-1]
    toppoints = interpolate_path(np.array([p7,p6,p5]), numVertical)
    toppoints = toppoints[1:-1]
    vertical_traces = [[bottompoints[i], newmiddle_points[i], toppoints[i]] for i in range(numVertical-2)]
    return horizontalTraces, vertical_traces

def construct_final_traces(fingerRegionDict, horizontalPalm, verticalPalm):
    final_horizontal = []
    final_vertical = []
    for i in range(4):
        trace = np.vstack([horizontalPalm[i], fingerRegionDict['t2'][3][i][::-1], fingerRegionDict['t1'][3][i][::-1]])
        final_horizontal.append(trace)
    for i in range(11):
        if i == 0:
            trace = np.vstack([verticalPalm[i],fingerRegionDict['d5p3'][3][1][::-1],fingerRegionDict['d5p2'][3][1][::-1],fingerRegionDict['d5p1'][3][1][::-1]])
        elif i == 1:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d5p3'][3][0][::-1], fingerRegionDict['d5p2'][3][0][::-1], fingerRegionDict['d5p1'][3][0][::-1]])
        elif i == 3:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d4p3'][3][1][::-1], fingerRegionDict['d4p2'][3][1][::-1], fingerRegionDict['d4p1'][3][1][::-1]])
        elif i == 4:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d4p3'][3][0][::-1], fingerRegionDict['d4p2'][3][0][::-1] , fingerRegionDict['d4p1'][3][0][::-1]])
        elif i == 6:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d3p3'][3][1][::-1], fingerRegionDict['d3p2'][3][1][::-1], fingerRegionDict['d3p1'][3][1][::-1]])
        elif i == 7:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d3p3'][3][0][::-1], fingerRegionDict['d3p2'][3][0][::-1], fingerRegionDict['d3p1'][3][0][::-1]])
        elif i == 9:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d2p3'][3][1][::-1], fingerRegionDict['d2p2'][3][1][::-1], fingerRegionDict['d2p1'][3][1][::-1]])
        elif i == 10:
            trace = np.vstack([verticalPalm[i], fingerRegionDict['d2p3'][3][0][::-1], fingerRegionDict['d2p2'][3][0][::-1], fingerRegionDict['d2p1'][3][0][::-1]])
        else:
            trace = np.array(verticalPalm[i])
        final_vertical.append(trace)
    return (final_horizontal, final_vertical)

def extrude_contour(vertices, extrusion_distance):
    """
    Extrudes the contour outward by a given distance based on local normals.
    :param vertices: Nx2 array of contour vertices (x, y).
    :param extrusion_distance: Distance to extrude outward.
    :return: Extruded contour vertices (Nx2 array).
    """
    vertices = np.asarray(vertices)
    num_vertices = len(vertices)

    extruded_vertices = []
    for i in range(num_vertices):
        prev_vertex = vertices[i - 1]
        curr_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % num_vertices]  # Wrap around

        tangent1 = curr_vertex - prev_vertex
        tangent2 = next_vertex - curr_vertex

        tangent1 /= np.linalg.norm(tangent1)
        tangent2 /= np.linalg.norm(tangent2)

        average_tangent = tangent1 + tangent2
        average_tangent /= np.linalg.norm(average_tangent)

        normal = np.array([-average_tangent[1], average_tangent[0]])

        extruded_vertex = curr_vertex + extrusion_distance * normal
        extruded_vertices.append(extruded_vertex)

    return np.array(extruded_vertices)

def line_intersection(p1, p2, p3, p4):
    """
    Calculate the intersection point of two line segments (p1-p2) and (p3-p4).
    :param p1, p2: Endpoints of the first line segment.
    :param p3, p4: Endpoints of the second line segment.
    :return: Intersection point (x, y) or None if no intersection.
    """
    # Line p1-p2 represented as a1x + b1y = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]
    
    # Line p3-p4 represented as a2x + b2y = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]
    
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        return None  # Lines are parallel or coincident
    
    # Compute the intersection point
    intersect_x = (b2 * c1 - b1 * c2) / determinant
    intersect_y = (a1 * c2 - a2 * c1) / determinant
    
    # Check if the intersection point is within the bounds of both segments
    if (min(p1[0], p2[0]) <= intersect_x <= max(p1[0], p2[0]) and
        min(p1[1], p2[1]) <= intersect_y <= max(p1[1], p1[1]) and
        min(p3[0], p4[0]) <= intersect_x <= max(p3[0], p4[0]) and
        min(p3[1], p4[1]) <= intersect_y <= max(p3[1], p4[1])):
        return np.array([intersect_x, intersect_y])
    
    return None

def remove_self_intersections(vertices):
    """
    Iteratively removes self-intersections by replacing the intersection points with the intersection itself.
    :param vertices: Nx2 array of contour vertices (x, y).
    :return: Cleaned contour vertices (Nx2 array).
    """
    num_vertices = len(vertices)
    new_vertices = vertices
    intersections_found = True
    
    while intersections_found:
        intersections_found = False
        for i in range(num_vertices):
            for j in range(i+2, num_vertices - (i == 0)):  # Avoid adjacent edges and wrap around
                p1, p2 = vertices[i], vertices[(i + 1) % num_vertices]
                p3, p4 = vertices[j], vertices[(j + 1) % num_vertices]
                
                intersection = line_intersection(p1, p2, p3, p4)
                if intersection is not None:
                    # Replace the intersecting vertices with the intersection point
                    new_vertices[i] = intersection
                    new_vertices[(i + 1) % num_vertices] = intersection
                    intersections_found = True
                    break  # Stop checking other intersections once one is found
            if intersections_found:
                break  # Stop checking if we found an intersection
    
        # Update vertices for the next iteration
        vertices = np.array(new_vertices)
    
    return np.array(new_vertices)

def intrude_contour(vertices, extrusion_distance):
    """
    Extrudes the contour outward by a given distance based on local normals,
    and removes any self-intersections by replacing them with the intersected point itself.
    :param vertices: Nx2 array of contour vertices (x, y).
    :param extrusion_distance: Distance to extrude outward.
    :return: Extruded and intersection-free contour vertices (Nx2 array).
    """
    vertices = np.asarray(vertices)
    num_vertices = len(vertices)

    # Step 1: Extrude the contour
    extruded_vertices = []
    for i in range(num_vertices):
        prev_vertex = vertices[i - 1]
        curr_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % num_vertices]  # Wrap around

        tangent1 = curr_vertex - prev_vertex
        tangent2 = next_vertex - curr_vertex

        # Check if the tangent vectors have zero length
        norm_tangent1 = np.linalg.norm(tangent1)
        norm_tangent2 = np.linalg.norm(tangent2)

        if norm_tangent1 == 0 or norm_tangent2 == 0:
            continue  # Skip this iteration if a tangent has zero length

        # Normalize the tangent vectors
        tangent1 /= norm_tangent1
        tangent2 /= norm_tangent2

        # Average tangent
        average_tangent = tangent1 + tangent2
        average_tangent /= np.linalg.norm(average_tangent)  # Normalize the average tangent
        
        normal = np.array([-average_tangent[1], average_tangent[0]])  # Perpendicular to the tangent
        
        extruded_vertex = curr_vertex - extrusion_distance * normal
        extruded_vertices.append(extruded_vertex)

    # Step 2: Remove self-intersections iteratively
    extruded_vertices = remove_self_intersections(extruded_vertices)

    return np.array(extruded_vertices)

def extrude_concave_shape(vertices, extrusion_distance, resolution=None, num_points=None):
    """
    Extrudes a concave shape by a specified distance.
    :param vertices: Nx2 array of contour vertices (x, y).
    :param extrusion_distance: Distance to extrude outward.
    :return: Extruded contour vertices (Nx2 array).
    """
    # Convert the vertices into a Shapely Polygon
    polygon = shapPolygon(vertices)

    # Perform the buffer operation for extrusion
    extruded_polygon = polygon.buffer(extrusion_distance)
    lower_left = Point(vertices[0])

    # If the polygons are disconnected, compute the concave hull (i.e. connect them)
    # shapely.concave_hull(extruded_polygon)

    nearest_pt = Point(min(extruded_polygon.exterior.coords, 
                  key=lambda x: Point(x).distance(lower_left)))
    # Get the exterior coordinates of the extruded shape
    # extruded_vertices = np.array(extruded_polygon.exterior.coords)

    perimeter = extruded_polygon.exterior.coords
    new_coords = []
    first_vertex = nearest_pt  # as found in the question above
    print(first_vertex)
    two_tours = itertools.chain(perimeter[:-1], perimeter)
    for v in two_tours:
        if Point(v) == first_vertex:
            new_coords.append(v)
            while len(new_coords) < len(perimeter):
                new_coords.append(next(two_tours))
            break
    polygon = shapPolygon(new_coords)
    # extruded_vertices = np.array(polygon.exterior.coords)

    if resolution:
        # Generate points based on fixed resolution
        num_points = int(polygon.exterior.length / resolution)
    elif not num_points:
        # Default to using the number of input vertices if no resolution or num_points provided
        num_points = len(vertices)

    # Interpolate points along the exterior
    uniform_coords = [
        polygon.exterior.interpolate(i / num_points, normalized=True).coords[0]
        for i in range(num_points)
    ]

    # Ensure it forms a closed loop if necessary
    if uniform_coords[0] != uniform_coords[-1]:
        uniform_coords.append(uniform_coords[0])

    return np.array(uniform_coords)

def extrude_outward(mask, extrusion_distance):
    """
    Extrude a binary mask outward by a specified distance.
    :param mask: 2D numpy array (binary mask where 1s represent the hand).
    :param extrusion_distance: Distance to extrude outward (in pixels).
    :return: 2D numpy array of the outward-extruded mask.
    """
    kernel_size = int(extrusion_distance * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    extruded_mask = cv2.dilate(mask, kernel)
    return extruded_mask

def extrude_inward(mask, extrusion_distance):
    """
    Extrude a binary mask inward by a specified distance.
    :param mask: 2D numpy array (binary mask where 1s represent the hand).
    :param extrusion_distance: Distance to extrude inward (in pixels).
    :return: 2D numpy array of the inward-extruded mask.
    """
    kernel_size = int(extrusion_distance * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    extruded_mask = cv2.erode(mask, kernel)
    return extruded_mask

def create_coverlay_opening(edge, distance):
    return extrude_concave_shape(edge, -distance)

def point_to_segment_distance_np(points, A, B):
    """Calculate the minimum distance from points to a line segment AB."""
    A = np.asarray(A)
    B = np.asarray(B)
    points = np.atleast_2d(points)  # Ensure 2D array

    AB = B - A
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        # A and B are the same point
        return np.linalg.norm(points - A, axis=1)

    # Parametric t for projection on AB
    AP = points - A
    t = np.clip(np.sum(AP * AB, axis=1) / AB_squared, 0, 1)
    projections = A + t[:, np.newaxis] * AB

    return np.linalg.norm(points - projections, axis=1)

def is_below_segment(points, A, B):
    """Check if points are 'below' the line segment AB using the cross product."""
    A = np.asarray(A)
    B = np.asarray(B)
    points = np.atleast_2d(points)

    AB = B - A
    AP = points - A

    # Cross product in 2D: z = x1*y2 - y1*x2
    cross_products = AB[0] * AP[:, 1] - AB[1] * AP[:, 0]
    return cross_products < 0  # True if 'below'

def closest_point_with_condition_np(track, point, A, B, min_distance=0.7):
    """Find the closest point on the track to 'point' that is at least
    'min_distance' below the line segment AB."""
    track = np.asarray(track[:len(track)//2])  # Use first half of the track
    distances_to_segment = point_to_segment_distance_np(track, A, B)
    below_segment_mask = is_below_segment(track, A, B)
    below_point_mask = track[:,1]>point[1]
    valid_points = track[(distances_to_segment >= min_distance) & below_segment_mask & below_point_mask]

    if valid_points.size == 0:
        return None  # No point satisfies the condition

    distances_to_point = np.linalg.norm(valid_points - point, axis=1)
    return valid_points[np.argmin(distances_to_point)]

def sample_track(fingerRegionDict, handEdge, boundary, horizontalPalm, verticalPalm, smoothedEdge, extrusionDist=2):
    horizontalTraces = []
    intrudedContours = []
    intrudedContours2 = []
    thisEdge = smoothedEdge[::-1]
    thisEdge = thisEdge[:, [1, 0]]

    coverlay = create_coverlay_opening(thisEdge,distance=16*extrusionDist)

    for i in range(16):
        nextContour = extrude_concave_shape(thisEdge, -extrusionDist*i)
        intrudedContours.append(nextContour.copy())
    for i in range(16):
        nextContour = extrude_concave_shape(smoothedEdge[:,[1,0]], -extrusionDist*i)
        intrudedContours2.append(nextContour.copy())

    for i in range(9):
        # v2_inner = extrude_inward(handEdge, 0.7*(8-i))
        # inner_target_l, _ = contour_hand(v2_inner, boundary)
        inner_target_l = intrudedContours[8-i]
        revisedHorizontalIndex = 8-i
        # inner_target_l = inner_target_l[::-1]
        track=inner_target_l
        # if i == 0:
        #     ax3.scatter(track[:,0],track[:,1], color='magenta')
        for j in range(5,1,-1):
            patch = 'd'+str(j)+'p'+ str(3-i//3)
            electrode = i%3
            two_points = np.array(fingerRegionDict[patch][2][::-1][electrode])

            # find closest point on inner_target_l to each point, and create new track where each point in two_points
            # is connected to its closest point, and all points in between the two closest points are removed
            # (ie, closestPoint1, twoPoints1, twoPoints2, closestPoint2)
            # Find the closest points on `inner_target_l` to each point in `two_points`
            closest_points = []
            for point in two_points:
                closest_point = min(track, key=lambda p: euclidean_distance(point, p))
                closest_points.append(closest_point)
            if i == 0 and j == 5:
                print(closest_points)
            
            # Get indices of the closest points in `inner_target_l`
            closest_indices = [np.where((track == cp).all(axis=1))[0][0] for cp in closest_points]
            closest_indices.sort()  # Sort indices to maintain order
            
            idx_start, idx_end = closest_indices
            if j==2: 
                track = np.concatenate((track[:idx_start + 1], two_points[::-1]))
            else:
                track = np.concatenate((track[:idx_start + 1], two_points[::-1], track[idx_end:]))
        horizontalTraces.append((track,revisedHorizontalIndex))

    #Connect horizontal traces on palm
    horizontalOrdered = np.array(horizontalPalm)[::-1]
    for i in range(9, 16):
        # v2_inner = extrude_inward(handEdge, 0.7*i)
        # inner_target_l, _ = contour_hand(v2_inner, boundary)
        inner_target_l = intrudedContours[i]
        # inner_target_l = inner_target_l[::-1]
        track=inner_target_l
        point = horizontalOrdered[i-9][0]
        closest_point = min(track[:len(track)//2], key=lambda p: euclidean_distance(point, p))
        closestIdx = np.where((track == closest_point).all(axis=1))[0][0]
        if i >= 12:
            track = np.concatenate((track[:closestIdx], horizontalOrdered[i-9],  fingerRegionDict['t2'][3][15-i][::-1], fingerRegionDict['t1'][3][15-i][::-1]))
        else:
            point2 = horizontalOrdered[i-9][-1]
            closest_2 = min(coverlay, key=lambda p: euclidean_distance(point2, p))
            track = np.vstack((track[:closestIdx], horizontalOrdered[i-9], closest_2))
        horizontalTraces.append((track,i))

    verticalTraces = []
    #Vertical thumb
    for i in range(0,5):
        # v2_inner = extrude_inward(handEdge, 0.7*i)
        # inner_target_l, _ = contour_hand(v2_inner, boundary)

        inner_target_l = intrudedContours2[i]
        track=inner_target_l[::-1]
        if i < 3:
            electrode = fingerRegionDict['t1'][2][i]
        else:
            electrode = fingerRegionDict['t2'][2][i-3]
        point = electrode[0]
        closest_point = min(track, key=lambda p: euclidean_distance(point, p))
        closestIdx = np.where((track == closest_point).all(axis=1))[0][0]
        track = np.concatenate((track[:closestIdx], electrode))
        verticalTraces.append((track,i))
    for i in range(10, -1, -1):
        inner_target_l = intrudedContours2[15-i]
        track=inner_target_l[::-1]

        point = verticalPalm[i][0]
        #Look only for point on the left part of the hand
        if i > 5:
            closest_point = min(track[:len(track)//2], key=lambda p: euclidean_distance(point, p))
            prevA, prevB = point, closest_point
        else:
            closest_point = closest_point_with_condition_np(track, point, prevA,prevB, min_distance=8)
            prevA, prevB = point, closest_point
        
        closestIdx = np.where((track == closest_point).all(axis=1))[0][0]
        if i == 0:
            trace = np.vstack([track[:closestIdx],verticalPalm[i],fingerRegionDict['d5p3'][3][1][::-1],fingerRegionDict['d5p2'][3][1][::-1],fingerRegionDict['d5p1'][3][1][::-1]])
        elif i == 1:
            trace = np.vstack([track[:closestIdx],verticalPalm[i], fingerRegionDict['d5p3'][3][0][::-1], fingerRegionDict['d5p2'][3][0][::-1], fingerRegionDict['d5p1'][3][0][::-1]])
        elif i == 3:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d4p3'][3][1][::-1], fingerRegionDict['d4p2'][3][1][::-1], fingerRegionDict['d4p1'][3][1][::-1]])
        elif i == 4:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d4p3'][3][0][::-1], fingerRegionDict['d4p2'][3][0][::-1] , fingerRegionDict['d4p1'][3][0][::-1]])
        elif i == 6:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d3p3'][3][1][::-1], fingerRegionDict['d3p2'][3][1][::-1], fingerRegionDict['d3p1'][3][1][::-1]])
        elif i == 7:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d3p3'][3][0][::-1], fingerRegionDict['d3p2'][3][0][::-1], fingerRegionDict['d3p1'][3][0][::-1]])
        elif i == 9:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d2p3'][3][1][::-1], fingerRegionDict['d2p2'][3][1][::-1], fingerRegionDict['d2p1'][3][1][::-1]])
        elif i == 10:
            trace = np.vstack([track[:closestIdx], verticalPalm[i], fingerRegionDict['d2p3'][3][0][::-1], fingerRegionDict['d2p2'][3][0][::-1], fingerRegionDict['d2p1'][3][0][::-1]])
        else:
            point2 = verticalPalm[i][-1]
            closest_2 = min(coverlay, key=lambda p: euclidean_distance(point2, p))
            trace = np.vstack((track[:closestIdx], verticalPalm[i], closest_2))
        verticalTraces.append((trace,15-i))
    
    #connect convex hull of the sensing patches to the innermost hand outline extrusion for the coverlay opening
    for j in range(5,0,-1):
        if j!=1:
            patch = 'd'+str(j)+'p'
            bottomRight = fingerRegionDict[patch+"3"][1][2]
            bottomLeft = fingerRegionDict[patch+"3"][1][3]
            closestPointRight = min(coverlay, key=lambda p: euclidean_distance(bottomRight, p))
            closestPointLeft = min(coverlay, key=lambda p: euclidean_distance(bottomLeft, p))
            closestLeft = np.where((coverlay == closestPointLeft).all(axis=1))[0][0]
            closestRight = np.where((coverlay == closestPointRight).all(axis=1))[0][0]
            coverlay = np.vstack((coverlay[:closestRight+1], bottomRight, fingerRegionDict[patch+"3"][1][1],
                                    fingerRegionDict[patch+"2"][1][2], fingerRegionDict[patch+"2"][1][1],
                                    fingerRegionDict[patch+"1"][1][2], fingerRegionDict[patch+"1"][1][1],
                                    fingerRegionDict[patch+"1"][1][0], fingerRegionDict[patch+"1"][1][3],
                                    fingerRegionDict[patch+"2"][1][0], fingerRegionDict[patch+"2"][1][3],
                                    fingerRegionDict[patch+"3"][1][0], bottomLeft, coverlay[closestLeft:]))
        else:
            patch = 't'
            bottomRight = fingerRegionDict[patch+"2"][1][2]
            bottomLeft = fingerRegionDict[patch+"2"][1][3]
            closestPointRight = min(coverlay, key=lambda p: euclidean_distance(bottomRight, p))
            closestPointLeft = min(coverlay, key=lambda p: euclidean_distance(bottomLeft, p))
            closestLeft = np.where((coverlay == closestPointLeft).all(axis=1))[0][0]
            closestRight = np.where((coverlay == closestPointRight).all(axis=1))[0][0]
            coverlay = np.vstack((coverlay[:closestRight+1], bottomRight, fingerRegionDict[patch+"2"][1][1],
                                    fingerRegionDict[patch+"1"][1][2], fingerRegionDict[patch+"1"][1][1],
                                    fingerRegionDict[patch+"1"][1][0], fingerRegionDict[patch+"1"][1][3],
                                    fingerRegionDict[patch+"2"][1][0], bottomLeft, coverlay[closestLeft:]))
    horizontal_traces = [sortedTrace[0] for sortedTrace in sorted(horizontalTraces, key=lambda x:x[1])]
    vertical_traces = [sortedTrace[0] for sortedTrace in sorted(verticalTraces, key=lambda x:x[1])]
    return horizontal_traces, vertical_traces, coverlay

def extend_tracks_to_via(contour, horizontal, vertical, startDistance=0.6*app.pixels_per_cm_height, startWidth=0.2*app.pixels_per_cm_height):
    # Note: changed startDistance from 0.7*app.pixels_per_cm_height to 0.6*app.pixels_per_cm_height
    viaSpacing = 0.05 * app.pixels_per_cm_width

    def get_point_along_contour(contour, distance, from_start=True):
        """Get a point along the contour at a specified distance from start or end."""
        segment_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        total_length = cumulative_lengths[-1]

        if distance > total_length:
            raise ValueError("Distance exceeds the contour length.")

        if from_start:
            idx = np.searchsorted(cumulative_lengths, distance)
            prev_length = cumulative_lengths[idx - 1] if idx > 0 else 0
            t = (distance - prev_length) / segment_lengths[idx]
            return contour[idx] + t * (contour[idx + 1] - contour[idx])
        else:
            reversed_lengths = total_length - cumulative_lengths
            idx = np.searchsorted(reversed_lengths[::-1], distance)
            prev_length = reversed_lengths[-(idx + 1)] if idx > 0 else 0
            t = (distance - prev_length) / segment_lengths[-(idx + 1)]
            return contour[-(idx + 2)] + t * (contour[-(idx + 1)] - contour[-(idx + 2)])

    # Get the starting points along the contour
    startContour = get_point_along_contour(contour, startDistance, from_start=True)
    endContour = get_point_along_contour(contour, startDistance, from_start=False)

    # Generate new via points for vertical tracks
    vertical_vias = np.array([
        startContour + np.array([startWidth + i * viaSpacing, 0])
        for i in range(len(vertical))
    ])

    # Generate new via points for horizontal tracks
    horizontal_vias = np.array([
        endContour + np.array([-startWidth - i * viaSpacing, 0])
        for i in range(len(horizontal))
    ])

    # Connect each vertical track to its via point
    vertical_tracks = []
    for track, via_point in zip(vertical, vertical_vias):
        vertical_tracks.append(np.vstack([np.array([via_point[0],via_point[1]+viaSpacing]),via_point,track]))

    # Connect each horizontal track to its via point
    horizontal_tracks = []
    for track, via_point in zip(horizontal, horizontal_vias):
        horizontal_tracks.append(np.vstack([np.array([via_point[0],via_point[1]+viaSpacing]),via_point,track]))

    return {
        'vertical_tracks': vertical_tracks,
        'horizontal_tracks': horizontal_tracks,
        'vertical_vias': vertical_vias,
        'horizontal_vias': horizontal_vias
    }

#Fix routing to horizontal thumb traces
#Wider top thumb patch


def scale_coordinates(points, pixels_per_cm_width, pixels_per_cm_height):
    """
    Scale a list of (x, y) points from pixels to centimeters.

    Args:
        points (list of tuples): List of (x, y) tuples representing points in pixels.
        pixels_per_cm_width (float): Conversion factor from pixels to cm along the width.
        pixels_per_cm_height (float): Conversion factor from pixels to cm along the height.

    Returns:
        list of tuples: List of (x, y) tuples representing the scaled points in centimeters.
    """
    return [(x*(pixels_per_cm_width), y*(pixels_per_cm_height)) for x, y in points]

def create_outline_path(points, thickness, dwg):
    """
    Create an SVG with a true outline of a path.
    
    Args:
        points (list of tuples): List of (x, y) points defining the original path.
        thickness (float): Thickness of the outline.
        dwg (svgwrite.Drawing): The SVG drawing object.
        
    Returns:
        outline_polygon (svgwrite.Polygon): The created outline path and outline (geometry)
    """
    # Create the original path as a LineString
    path = LineString(points)
    
    # Buffer the path to create an outline (shapely handles offsetting)
    outline = path.buffer(thickness / 2, cap_style=2, join_style=2)  # Thickness/2 because buffer expands both sides

    # Ensure the outline is oriented correctly
    outline = orient(outline, sign=1.0)
    
    # Extract exterior points from the buffered polygon
    if not isinstance(outline, shapPolygon):
        raise ValueError("Buffer operation failed; the result is not a Polygon.")
    
    exterior_points = list(outline.exterior.coords)
    # Add the outline polygon to the SVG
    outline_polygon = dwg.polygon(points=exterior_points, fill="black", stroke="none")
    return outline_polygon, outline

def add_polygon_to_svg(dwg, group, polygon, stroke="none", fill="black"):
    coords = list(polygon.exterior.coords)
    group.add(dwg.polygon(points=coords, stroke=stroke, fill=fill))

def save_paths(image_id, image, vertical_traces, horizontal_traces, edgeCut, coverlaymask, finger_rectangles, palmRegion):
    """
    Save multiple paths in a grouped structure in an SVG file, scaling from pixels to centimeters.
    
    Args:
        image (ndarray): The image array for setting SVG size.
        vertical_traces (list of list of tuples): List of (x, y) tuples for vertical paths.
        horizontal_traces (list of list of tuples): List of (x, y) tuples for horizontal paths.
        edgeCut (list of tuples): List of (x, y) points for the edge cut polygon.
        coverlaymask (list of tuples): List of (x, y) points for the coverlay mask.
        pixels_per_cm_width (float): Conversion factor from pixels to cm along the width.
        pixels_per_cm_height (float): Conversion factor from pixels to cm along the height.
    """
    newScaleY = 11*96/image.shape[0]
    newScaleX = 8.5*96/image.shape[1]
    print(newScaleY, newScaleX)
    # dwgFront = svgwrite.Drawing("handFront.svg")
    # dwgBack = svgwrite.Drawing("handBack.svg")
    front_svg_path = os.path.join("output", f"{image_id}_front.svg")
    back_svg_path = os.path.join("output", f"{image_id}_back.svg")
    dwgFront = svgwrite.Drawing(front_svg_path)
    dwgBack = svgwrite.Drawing(back_svg_path)

    thickness = 0.5  # Scaling factor for line thickness
    frontCopperPolygons=[]
    backCopperPolygons=[]

    # Create groups for the paths
    frontCopper = dwgFront.g(id="F.Cu", fill="black", stroke="none")
    backCopper = dwgBack.g(id="B.Cu", fill="black", stroke="none")
    frontcuts = dwgFront.g(id="Edge.Cuts", fill="gray", stroke="none")
    backCuts = dwgBack.g(id="Edge.Cuts", fill="gray", stroke="none")
    frontmask = dwgFront.g(id="F.Mask", fill="black", stroke="none")
    backmask = dwgBack.g(id="B.Mask", fill="black", stroke="none")
    adhesiveFront = dwgFront.g(id="Eco1.User", fill="black", stroke="none")
    adhesiveBack = dwgBack.g(id="Eco1.User", fill="black", stroke="none")

    # Scale and add each vertical trace after outlining
    for i, path_points in enumerate(vertical_traces):
        #7, 10, 13
        path_points = scale_coordinates(path_points, newScaleX, newScaleY)
        fullPath, outlinePolygon = create_outline_path(path_points, thickness, dwgFront)
        if i in [7, 10, 13]:
            outlinedPath,_ = create_outline_path(path_points[:-1], thickness, dwgFront)
        else:
            outlinedPath = fullPath
        frontCopperPolygons.append(outlinePolygon)
        frontCopper.add(outlinedPath)

    # Scale and add each horizontal trace after outlining
    for i, path_points in enumerate(horizontal_traces):
        #9, 10, 11
        path_points = scale_coordinates(path_points, newScaleX, newScaleY)
        fullPath,outlinePolygon = create_outline_path(path_points, thickness, dwgBack)
        if i in [9,10,11]:
            outlinedPath,_ = create_outline_path(path_points[:-1], thickness, dwgBack)
        else:
            outlinedPath = fullPath
        backCopperPolygons.append(outlinePolygon)
        backCopper.add(outlinedPath)

    for polygon_points in finger_rectangles:
        maskPolygon = dwgFront.polygon(
            points=scale_coordinates(polygon_points, newScaleX, newScaleY),
            fill='black', stroke='none'
        )
        frontmask.add(maskPolygon)
        maskPolygon2 = dwgBack.polygon(
            points=scale_coordinates(polygon_points, newScaleX, newScaleY),
            fill='black', stroke='none'
        )
        backmask.add(maskPolygon2)

    # Scale and add the coverlay mask for the palm
    frontmaskLayer = dwgFront.polygon(
        points=scale_coordinates(palmRegion, newScaleX,newScaleY),
        fill='black', stroke='none')
    frontmask.add(frontmaskLayer)

    backmaskLayer = dwgBack.polygon(
        points=scale_coordinates(palmRegion, newScaleX, newScaleY),
        fill='black', stroke='none'
    )
    backmask.add(backmaskLayer)

    # Scale and add the coverlay mask
    # frontmaskLayer = dwgFront.polygon(
    #     points=scale_coordinates(coverlaymask, newScaleX,newScaleY),
    #     fill='black', stroke='none')
    # frontmask.add(frontmaskLayer)

    # backmaskLayer = dwgBack.polygon(
    #     points=scale_coordinates(coverlaymask, newScaleX, newScaleY),
    #     fill='black', stroke='none'
    # )
    # backmask.add(backmaskLayer)

    #Step 1: Join cyan and green polygons
    all_inner_polygons = unary_union(frontCopperPolygons + backCopperPolygons)
    maskPolygon = shapPolygon(scale_coordinates(coverlaymask, newScaleX,newScaleY))
    resulting_polygon = maskPolygon.difference(all_inner_polygons)
    innerPolygons = resulting_polygon.buffer(-1.75)
    print(type(innerPolygons))

    scaledEdge = scale_coordinates(edgeCut, newScaleX, newScaleY)


    # Scale and add the edge cut polygons
    edgeFront = dwgFront.polygon(
        points=scaledEdge,
        fill='black', stroke='none'
    )
    edgeBack = dwgBack.polygon(
        points=scaledEdge,
        fill='black', stroke='none'
    )


    frontcuts.add(edgeFront)
    backCuts.add(edgeBack)
    print("Added edgeFront and edgeBack polygons!")
    for extrudedInner in innerPolygons.geoms:
        print(type(extrudedInner))
        add_polygon_to_svg(dwgFront,frontcuts,extrudedInner)
        add_polygon_to_svg(dwgBack,backCuts,extrudedInner)
    print(scaledEdge[0],scaledEdge[-1])
    print(maskPolygon.exterior.coords[0],maskPolygon.exterior.coords[-1])
    edgePolygon = shapPolygon(scaledEdge)
    print(edgePolygon.exterior.coords[0], edgePolygon.exterior.coords[-1])

    try:
        result = edgePolygon.difference(maskPolygon)
        print("Success!")
    except:
        print("WARNING: Geometries might be invalid. Consider redrawing traces.")
        # Use buffer(0) to prevent ivalid geometries that can cause a TopologyException
        cleaned_edgePolygon = edgePolygon.buffer(0)
        cleaned_maskPolygon = maskPolygon.buffer(0)

        # result = edgePolygon.difference(maskPolygon)
        result = cleaned_edgePolygon.difference(cleaned_maskPolygon)

    path_data = ""
    polygons_to_draw = []

    # Check if the result is a single Polygon or a MultiPolygon
    if result.geom_type == 'Polygon':
        polygons_to_draw.append(result)
    elif result.geom_type == 'MultiPolygon':
        polygons_to_draw.extend(list(result.geoms))
    
    # Loop through the list of polygons to draw
    for poly in polygons_to_draw:
        # Build the path data string for ONLY this single polygon and its holes
        path_data_segment = "M " + " L ".join(f"{x},{y}" for x, y in poly.exterior.coords) + " Z"
        for hole in poly.interiors:
            path_data_segment += " M " + " L ".join(f"{x},{y}" for x, y in hole.coords) + " Z"

        # Create and add a NEW <path> element for just this one segment.
        # This is much more efficient and avoids the giant string problem.
        adhesiveFront.add(dwgFront.path(d=path_data_segment, fill="black", stroke="none"))
        adhesiveBack.add(dwgBack.path(d=path_data_segment, fill="black", stroke="none"))

    # # Now, build the SVG path data by iterating through all polygons.
    # # This loop works correctly whether there is one polygon or many.
    # for poly in polygons_to_draw:
    #     # Add the exterior boundary path
    #     path_data += " M " + " L ".join(f"{x},{y}" for x, y in poly.exterior.coords) + " Z"
    #     # Add the paths for any interior holes
    #     for hole in poly.interiors:
    #         path_data += " M " + " L ".join(f"{x},{y}" for x, y in hole.coords) + " Z"

    # # Add the path to the SVG
    # # Note: potential typo fixed? Changed dwgFront to dwgBack for adhesiveBack
    # adhesiveFront.add(dwgFront.path(d=path_data, fill="black", stroke="none"))
    # adhesiveBack.add(dwgBack.path(d=path_data, fill="black", stroke="none"))

    # Add the groups to the SVGs
    dwgFront.add(frontCopper)
    dwgFront.add(frontcuts)
    dwgFront.add(frontmask)
    dwgFront.add(adhesiveFront)

    dwgBack.add(backCopper)
    dwgBack.add(backCuts)
    dwgBack.add(backmask)
    dwgBack.add(adhesiveBack)

    # Save the SVG files
    dwgFront.save()
    dwgBack.save()

def create_kicad_mod(input_svg_path: str, output_kicad_mod_path: str):
    """
    Converts an SVG file to a .kicad_mod file.
    """
    if not os.path.exists(input_svg_path):
        raise FileNotFoundError(f"Required SVG file not found at {input_svg_path}")

    # Load the SVG file
    # for side in ["Front","Back"]:
        # tree = etree.parse(f"hand{side}.svg")
    tree = etree.parse(input_svg_path)
    root = tree.getroot()

    # Define namespaces
    INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"

    # Add inkscape:label to all <g> elements with an id
    for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
        group_id = g.get("id")
        if group_id:
            g.set(f"{{{INKSCAPE_NS}}}label", group_id)

    # Save to a new file
    new_svg_path = f"{input_svg_path[:-4]}_new.svg"
    tree.write(new_svg_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    # tree.write(f"hand{side}New.svg", pretty_print=True, xml_declaration=True, encoding="UTF-8")
    imported = Svg2ModImport(
        # f"output/hand{side}New.svg",
        new_svg_path,
        "svg2mod",
        "G***",
        True,
        None
    )

    # # Pick an output file name if none was provided:
    # output_file_name = f"hand{side}.kicad_mod"

    # Create an exporter:
    exported = Svg2ModExportLatest(
        imported,
        # output_file_name,
        output_kicad_mod_path,
        False,
        1.0,
        5.0,
        dpi = DEFAULT_DPI,
        pads = False,
    )

    cmd_args = [os.path.basename(sys.argv[0])] + sys.argv[1:]
    cmdline = ' '.join(shlex.quote(x) for x in cmd_args)

    # Export the footprint:
    exported.write(cmdline)


def add_vias(filename):
    via_location_tuples = []
    lines = ""

    with open(filename, 'r') as file:
        contents = file.read()

        # Get lines to calculate number of lines (to use when writing)
        file.seek(0) # Reset pointer
        lines = file.readlines()

        # Split contents by fp_poly sections
        sections = contents.split("fp_poly\n")

        for index, section in enumerate(sections[1:17]): # First section is title/header, so we can ignore it, and we need 16 vias
            parts = section.split()

            # This makes the parts variable a list of the different points
            # E.g. ['(pts', '(xy', '83.83730045586059', '239.78340137748694)', ... ]

            # Extract the x and y coordinates from each section
            via_x_coord = (float(parts[5])+float(parts[8]))/2 # Middle of the two x-coordinates of the ends of the line
            via_y_coord = float(parts[6][:-1]) # y-coordinate minus the ')'

            via_text = f'(pad "{index+1}" thru_hole circle\n(at {via_x_coord} {via_y_coord})\n(size 0.35 0.35)\n(drill 0.15)\n(layers "*.Cu" "*.Mask")\n(remove_unused_layers no)\n)\n'

            lines.insert(len(lines) - 1, via_text)

            via_location_tuples.append((via_x_coord, via_y_coord))

            # Write via to file!
        with open(filename, 'w') as file:
            file.writelines(lines)
        
        print(f"Via Location Tuples: {via_location_tuples}")

# --- End Global Helper Functions ---

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    image_data = await file.read()
    
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(io.BytesIO(image_data))

    # Convert PIL image to NumPy array (OpenCV uses NumPy arrays)
    open_cv_image = np.array(image)
    
    # Convert RGB (PIL) to BGR (OpenCV)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # ----- Crop Image (based on paper background)-----

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)

    # Define the range for white in HSV
    # lower_white = np.array([0, 0, 200])  # Minimum hue, saturation, and value
    # upper_white = np.array([250, 60, 255])  # Maximum hue, saturation, and value
    lower_white = np.array([0, 0, 120])  # Minimum hue, saturation, and value
    upper_white = np.array([250, 50, 255])  # Maximum hue, saturation, and value

    # Create a mask
    mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour (likely the paper)
    paper_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the paper
    x, y, w, h = cv2.boundingRect(paper_contour)

    # Crop the paper from the image
    cropped_paper = hsv_image[y:y+h, x:x+w]
    
    # Calculate the pixel-to-inch ratio based on known paper size (8.5x11 inches)
    paper_width_cm = 8.5 * 2.54  # Width of paper in cm
    paper_height_cm = 11.0 * 2.54  # Height of paper in cm

    # Get the pixel dimensions of the cropped paper
    pixel_height, pixel_width, _ = cropped_paper.shape
    
    # Calculate pixels per inch for both width and height
    app.pixels_per_cm_width = pixel_width / paper_width_cm
    app.pixels_per_cm_height = pixel_height / paper_height_cm
    print(f"pixels_per_cm_width: {app.pixels_per_cm_width}")
    print(f"pixels_per_cm_height: {app.pixels_per_cm_height}")

    final_crop = cv2.cvtColor(cropped_paper, cv2.COLOR_HSV2RGB)
    final_crop_image = Image.fromarray(final_crop)
    
    # Create a directory to store processed images
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a unique ID for the image and its associated files
    image_id = uuid.uuid4().hex
    cropped_image_path = os.path.join(output_dir, f"{image_id}_cropped_hand.png")
    final_crop_image.save(cropped_image_path)

    hsv_crop = cv2.cvtColor(final_crop, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 20], dtype="uint8")
    # upper_skin = np.array([20, 255, 255], dtype="uint8")
    upper_skin = np.array([12, 255, 255], dtype="uint8")
    skin_mask1 = cv2.inRange(hsv_crop, lower_skin, upper_skin)
    
    lower_skin2 = np.array([172, 30, 30], dtype = "uint8")
    upper_skin2 = np.array([180, 255, 255], dtype = "uint8")
    skin_mask2 = cv2.inRange(hsv_crop, lower_skin2, upper_skin2)
    
    combined_skin_mask = cv2.bitwise_or(skin_mask1,skin_mask2)
    contours, _ = cv2.findContours(combined_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise HTTPException(status_code=400, detail="No hand contour found.")

    hand_contour = max(contours, key=cv2.contourArea)
    contoured_image = final_crop.copy()
    cv2.drawContours(contoured_image, [hand_contour], -1, (0, 255, 0), 3)

    hand_contour_points = hand_contour.squeeze().tolist()
    
    x = [p[0] for p in hand_contour_points]
    y = [p[1] for p in hand_contour_points]
    tck, u = splprep([x, y], s=3000, k=3)

    num_upsampled_points = int(1 * len(hand_contour_points))
    uu = np.linspace(u[0], u[-1], num_upsampled_points)
    upsampled_hand_contour_points = splev(uu, tck)
    upsampled_hand_contour_points = list(zip(upsampled_hand_contour_points[0], upsampled_hand_contour_points[1]))
    
    result_image = Image.fromarray(contoured_image)
    result_image_path = os.path.join(output_dir, f"{image_id}_contoured_hand.png")
    result_image.save(result_image_path)

    return JSONResponse(content={"message": "Contour detection complete",
                                 "image_id": image_id,
                                 "contour_image": f"/output/{image_id}_contoured_hand.png",
                                 "contour_points": upsampled_hand_contour_points})

class ContourRequest(BaseModel):
    contour: List[List[float]]

@app.post("/api/save-contour/{image_id}")
def save_contour(image_id: str, request: ContourRequest):
    contour = np.array(request.contour, dtype=np.int32)

    cropped_image_path = os.path.join("output", f"{image_id}_cropped_hand.png")
    if not os.path.exists(cropped_image_path):
        raise HTTPException(status_code=404, detail="Cropped image not found.")

    image = cv2.imread(cropped_image_path)
    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    mask_path = os.path.join("output", f"{image_id}_mask.png")
    cv2.imwrite(mask_path, mask)

    return JSONResponse(content={"message": "Mask creation complete", "mask_path": f"/output/{image_id}_mask.png"})


@app.post("/upload-mask/{image_id}")
async def upload_mask(image_id: str, file: UploadFile = File(...)):
    save_path = os.path.join("output", f"{image_id}_uploaded_mask.png")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={"mask_path": f"/output/{image_id}_uploaded_mask.png"})

@app.post("/generate-traces/{image_id}")
async def generate_traces(image_id: str):
    mask_path = os.path.join("output", f"{image_id}_uploaded_mask.png")
    image_path = os.path.join("output", f"{image_id}_cropped_hand.png")

    trace_start_distance = 1

    mp_hands = mp.solutions.hands
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    orig_cropped_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_cropped_image, cv2.COLOR_BGR2RGB)

    result = hands.process(image)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    if mask is None or orig_cropped_image is None:
        raise HTTPException(status_code=400, detail="Missing required image or mask for trace generation.")
    
    h, w = mask.shape[0], mask.shape[1]
    image_shape = (h, w)

    hand_landmarks = None
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0] # Use the first detected hand
    else:
        raise HTTPException(status_code=400, detail="No hand landmarks detected in the image for trace generation.")
    
    # Get y-coordinate of wrist landmark (landmark 0) to define boundary
    # x_landmark = (hand_landmarks.landmark[0].x)
    y_landmark = (hand_landmarks.landmark[0].y)
    
    # Calculate boundary for contour_hand
    boundary = int(y_landmark * mask.shape[0] + app.pixels_per_cm_height * trace_start_distance)
    
    # Ensure boundary is within image dimensions
    boundary = min(boundary, mask.shape[0])
    if boundary <= 0: # Ensure boundary is at least 1 pixel
        boundary = 1

    smoothed_v2 = smooth_hand_mask(mask)
    outer_edge, _ = contour_hand(smoothed_v2, boundary, num_points=5000)
    
    # Handle cases where outer_edge_palm might be empty
    if outer_edge.size == 0:
        raise HTTPException(status_code=400, detail="Could not detect a valid outer edge for palm region creation.")

    # Serialize hand_landmarks: convert MediaPipe Landmark objects to raw dictionaries/lists
    # serialized_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
    serialized_landmarks = [(int(lm.x * image_shape[1]), int(lm.y * image_shape[0])) for lm in hand_landmarks.landmark]

    regionDict = get_finger_polygons(serialized_landmarks, image.shape)
    finger_rectangles = [regionDict[region][1] for region in regionDict.keys() if regionDict[region][1] is not None]
    
    palmRegion = create_palm_region(serialized_landmarks, regionDict, outer_edge[:,[1,0]], int(y_landmark * mask.shape[0]), image.shape, app.pixels_per_cm_width)

    all_polygons = []

    for rect in finger_rectangles:
        polygon = [[float(x), float(y)] for (x, y) in rect]
        all_polygons.append(polygon)

    palm_polygon = [[float(x), float(y)] for (x, y) in palmRegion]
    all_polygons.append(palm_polygon)

    overlay = orig_cropped_image.copy()
    for poly in all_polygons:
        pts = np.array(poly, dtype=np.int32) 
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    overlay_filename = f"{image_id}_traced_hand_overlay.png"
    overlay_path = os.path.join("output", overlay_filename)
    cv2.imwrite(overlay_path, overlay)

    serializable_polygons = all_polygons
    serializable_polygons = {
        key: value[1].tolist() for key, value in regionDict.items() if value[1] is not None
    }  
    serializable_polygons['palm'] = palmRegion.tolist() 

    # Serialize regionDict: convert NumPy arrays to lists of lists of floats for JSON compatibility
    serializable_region_dict = {}
    for key, value in regionDict.items():
        joints_list = [[float(p[0]), float(p[1])] for p in value[0]] # joints are list of numpy arrays
        rectangle_list = [[float(p[0]), float(p[1])] for p in value[1]] # rectangle is a numpy array of points
        serializable_region_dict[key] = (joints_list, rectangle_list)
    
    # Serialize outer_edge, palmRegion, and smoother_v2
    serializable_outer_edge = outer_edge.tolist()
    serializable_palm_region = palmRegion.tolist()
    serializable_smoothed_v2 = smoothed_v2.tolist()

    # Create a new list where each NumPy array (rectangle) is converted to a Python list
    serializable_finger_rectangles = [
        regionDict[region][1].tolist()  # Convert the NumPy array to a list
        for region in regionDict.keys()
        if regionDict[region][1] is not None
    ]

    context_data = {
        "image_id": image_id,
        "hand_landmarks": serialized_landmarks, # MediaPipe landmarks as raw dicts
        "image_shape": image_shape, # (height, width)
        "pixels_per_cm_width": app.pixels_per_cm_width,
        "pixels_per_cm_height": app.pixels_per_cm_height,
        "region_dict": serializable_region_dict, # Finger rectangles + their defining joints
        "outer_edge": serializable_outer_edge,   # Outer hand contour
        "boundary": boundary,                 # Y-coordinate boundary
        "palm_region": serializable_palm_region, # Palm polygon (8 points)
        "finger_rectangles": serializable_finger_rectangles,
        "smoothed_v2": serializable_smoothed_v2, # Smoothed mask as list
        "mask_height": mask.shape[0],
    }
    context_file_path = os.path.join(os.path.dirname(overlay_path), f"{image_id}_context.json")
    with open(context_file_path, "w") as f:
        json.dump(context_data, f)
    print(f"--- DEBUG: Context data saved to {context_file_path} ---")


    return JSONResponse(content={
        "trace_id": image_id,
        "overlay_path": f"/output/{overlay_filename}",
        "polygons": serializable_polygons # These are the outline polygons
    })

class TraceData(BaseModel):
    # polygons: List[List[List[float]]]
    polygons: Dict[str, List[List[float]]]

@app.post("/save-edited-traces/{image_id}")
async def save_edited_traces(image_id: str, trace_data: TraceData):
    traces_file_path = os.path.join("output", f"{image_id}_traces.json")
    try:
        with open(traces_file_path, "w") as f:
            json.dump(trace_data.polygons, f)
        return JSONResponse(content={"message": "Traces saved successfully", "traces_file": f"/output/{image_id}_traces.json"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save traces: {e}")



@app.get("/get-final-outline/{image_id}")
async def get_final_outline(image_id: str):
    context_file_path = os.path.join("output", f"{image_id}_context.json")
    if not os.path.exists(context_file_path):
        raise HTTPException(status_code=404, detail="Context data not found.")

    try:
        with open(context_file_path, "r") as f:
            context_data = json.load(f)

        # Load only the data needed for the outline calculation
        loaded_smoothed_v2_list = context_data.get('smoothed_v2')
        # We need the wrist landmark to help define the bottom edge of the board
        y_landmark_val = context_data.get('hand_landmarks')[0][1]
        loaded_mask_height = context_data.get('mask_height')
        
        # Convert to NumPy array
        loaded_smoothed_v2_np = np.array(loaded_smoothed_v2_list, dtype=np.uint8)
        
        # Note: Same logic as /generate-kicad endpoint
        outer_edge_spacing = 0.1 # Same value as /generate-kicad endpoint
        edgeCut = extrude_outward(loaded_smoothed_v2_np, outer_edge_spacing * app.pixels_per_cm_height)
        boundaryEdge = int(y_landmark_val * loaded_mask_height + app.pixels_per_cm_height * app.bottom_edge_distance)
        outer_target, l_inner_target = contour_hand(edgeCut, boundaryEdge, num_points=5000)
        
        # The contour_hand function returns coordinates that need to be swapped for drawing
        outer_target = outer_target[:,[1,0]]

        # Ensure the final shape is a closed loop for drawing
        final_outline_polygon = np.vstack((outer_target, outer_target[0, :]))

        # Return the outline points as JSON
        return JSONResponse(content={"outline": final_outline_polygon.tolist()})

    except Exception as e:
        # Add the full traceback to server logs
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate outline: {e}")



@app.post("/generate-kicad/{image_id}")
async def generate_kicad(image_id: str):
    traces_file_path = os.path.join("output", f"{image_id}_traces.json")
    context_file_path = os.path.join("output", f"{image_id}_context.json")
    kicad_output_dir = "output"

    if not os.path.exists(traces_file_path):
        raise HTTPException(status_code=404, detail="Traces file not found. Please save edited traces first.")
    if not os.path.exists(context_file_path):
        raise HTTPException(status_code=404, detail="Context data for traces not found. Please generate traces first.")

    try:
        with open(traces_file_path, "r") as f:
            # Note: polygons_data is a dictionary of form {'t1': [[...]], 't2': [[...]]}
            polygons_data = json.load(f) # These are the outlines (edited by user)
        
        with open(context_file_path, "r") as f: # Load context data
            context_data = json.load(f)
        
        loaded_region_dict_serializable = context_data.get('region_dict')
        if not loaded_region_dict_serializable:
            raise HTTPException(status_code=404, detail="region_dict not found in context data.")

        # Iterate over the user-edited polygons (which are now in a dictionary)
        # and update the corresponding region in the main context object.
        for region_id, edited_polygon in polygons_data.items():
            if region_id in loaded_region_dict_serializable:
                # Update the polygon part (index 1) of the tuple
                loaded_region_dict_serializable[region_id][1] = edited_polygon
            # If a key doesn't exist, ignore it.

        # Reconstruct necessary variables from context_data for trace generation
        loaded_landmarks_raw = context_data.get('hand_landmarks') 
        loaded_image_shape = tuple(context_data.get('image_shape'))
        loaded_pixels_per_cm_width = context_data.get('pixels_per_cm_width')
        loaded_pixels_per_cm_height = context_data.get('pixels_per_cm_height')
        loaded_outer_edge_list = context_data.get('outer_edge')
        loaded_boundary = context_data.get('boundary')
        loaded_palm_region_list = context_data.get('palm_region')
        # loaded_finger_rectangles = context_data.get('finger_rectangles')
        loaded_smoothed_v2_list = context_data.get('smoothed_v2')
        loaded_mask_height = context_data.get('mask_height')

        # Convert back to NumPy arrays where functions expect them
        loaded_outer_edge_np = np.array(loaded_outer_edge_list, dtype=np.float32)
        # loaded_palm_region_np = np.array(loaded_palm_region_list, dtype=np.float32)
        loaded_smoothed_v2_np = np.array(loaded_smoothed_v2_list, dtype=np.uint8)
        
        loaded_palm_region_np = polygons_data.get('palm')

        # # Convert each inner list (representing a rectangle) back into a NumPy array
        # loaded_finger_rectangles_np = [
        #     np.array(rect_list, dtype=np.float32)  # Convert to NumPy array with float32 dtype
        #     for rect_list in loaded_finger_rectangles
        #     if rect_list is not None  # Ensure no None entries if your original data allowed them
        # ]
        finger_keys = [key for key in polygons_data.keys() if 'palm' not in key]
        loaded_finger_rectangles_np = [
            np.array(polygons_data[key], dtype=np.float32) for key in finger_keys
        ]

        # Reconstruct regionDict for use with helper functions
        # loaded_region_dict_serializable = context_data.get('region_dict')
        loaded_region_dict = {}
        for key, value in loaded_region_dict_serializable.items():
            joints_np = np.array(value[0], dtype=np.float32)
            rectangle_np = np.array(value[1], dtype=np.float32)
            loaded_region_dict[key] = (joints_np, rectangle_np)

        # Determine current y_landmark from loaded_landmarks_raw to pass to create_palm_region if needed
        # Assuming landmark 0 is wrist and y is its y-coordinate
        y_landmark_val = loaded_landmarks_raw[0][1] 

        # Convert raw landmarks (from context) to (x,y) pixel tuples suitable for helper functions
        # loaded_landmarks_pixel_coords = [[lm['x'] * loaded_image_shape[1], lm['y'] * loaded_image_shape[0]] for lm in loaded_landmarks_raw]


        print(f"\n--- DEBUG: Loaded polygons_data (type: {type(polygons_data)}) ---")
        # print(f"polygons_data (first 2 entries): {polygons_data[:2]}...")
        print(f"--- DEBUG: Loaded context_data (keys): {context_data.keys()} ---")
        print("------------------------------------------------------------------")

        # --- Trace Generation Logic (From Jupyter Notebook) ---
        electrode_traces = []
        horizontal_traces = []
        electrode_traces_other = []
        vertical_traces = []
        fingerRegionDict = {}

        for region in list(loaded_region_dict.keys()):
            p = loaded_region_dict[region][1]
            if region == 't1':
                pHorizontal = sample_and_connect_polygon_sides(p, num_points=5)
                pVertical = sample_and_connect_polygon_other_sides(p, num_points=6)
            elif region == 't2':
                pHorizontal = sample_and_connect_polygon_sides(p, num_points=4)
                pVertical = sample_and_connect_polygon_other_sides(p, num_points=6)
            else:
                pHorizontal = sample_and_connect_polygon_sides(p, num_points=5)
                pVertical = sample_and_connect_polygon_other_sides(p, num_points=4)
            electrode_traces.append(pHorizontal)
            horizontal_traces += pHorizontal
            electrode_traces_other.append(pVertical)
            vertical_traces += pVertical
            fingerRegionDict[region] = (loaded_region_dict[region][0], loaded_region_dict[region][1], pHorizontal, pVertical)
        # colorCorrected = imageio.imread("croppedHand.jpg")
        # mp_draw.draw_landmarks(colorCorrected, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        horizontalPalm, verticalPalm = sample_traces_palm(loaded_palm_region_np)
        # polygon = patches.Polygon(palmRegion, closed=True, edgecolor='r', fill=False, linewidth=2)
        horizontalFinal, verticalFinal = construct_final_traces(fingerRegionDict, horizontalPalm, verticalPalm)
        
        min_trace_spacing = 0.022
        outer_edge_spacing = 0.1
        newTracks, verticalTracks, coverlay = sample_track(fingerRegionDict,loaded_smoothed_v2_np,loaded_boundary, horizontalPalm, verticalPalm, loaded_outer_edge_np, extrusionDist=min_trace_spacing*app.pixels_per_cm_height)
        edgeCut = extrude_outward(loaded_smoothed_v2_np, outer_edge_spacing*app.pixels_per_cm_height)
        boundaryEdge = int(y_landmark_val * loaded_mask_height+app.pixels_per_cm_height*app.bottom_edge_distance)
        outer_target, l_inner_target = contour_hand(edgeCut, boundaryEdge, num_points=5000)
        outer_target = outer_target[:,[1,0]]

        outDict = extend_tracks_to_via(outer_target, newTracks, verticalTracks)

        # outer_target = reflect_thumb(outer_target[:,[1,0]], red_joint, purple_joint)
        edgeCut = np.vstack((outer_target, outer_target[0, :]))

        # --- Final Drawing of all traces (paths) in SVG ---
        # save_paths(binary_image,outDict['vertical_tracks'], outDict['horizontal_tracks'], outer_target,coverlay, finger_rectangles, loaded_palm_region_np)
        save_paths(image_id, loaded_smoothed_v2_np,outDict['vertical_tracks'], outDict['horizontal_tracks'], outer_target,coverlay, loaded_finger_rectangles_np, loaded_palm_region_np)

        front_svg_path = os.path.join("output", f"{image_id}_front.svg")
        back_svg_path = os.path.join("output", f"{image_id}_back.svg")
        front_kicad_path = os.path.join("output", f"{image_id}_handFront.kicad_mod")
        back_kicad_path = os.path.join("output", f"{image_id}_handBack.kicad_mod")
        zip_archive_path = os.path.join("output", f"{image_id}_kicad_files.zip")

        # create_kicad_mod()
        create_kicad_mod(front_svg_path, front_kicad_path)
        add_vias(front_kicad_path)
        create_kicad_mod(back_svg_path, back_kicad_path)

        with zipfile.ZipFile(zip_archive_path, 'w') as zf:
            # Add the files to the zip archive with clean names
            zf.write(front_kicad_path, arcname='handFront.kicad_mod')
            zf.write(back_kicad_path, arcname='handBack.kicad_mod')
        
        return FileResponse(
            path=zip_archive_path,
            media_type="application/zip",
            filename=f"{image_id}_kicad_pcb.zip"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"An unexpected error occurred during KiCad generation for image ID {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate KiCad file: {e}")