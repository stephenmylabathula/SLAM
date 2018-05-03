from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Read Serialized Laser Data in Meters
def loadData(file):
    with open(file, 'r') as f:
        return pickle.loads(f.read())[:,0]/1000

# Convert Laser Data to Local Frame XY Coordinates
def getLocalCoordinatesFromLaserData(distances):
    num_points = distances.shape[0]
    angle_scale_factor = num_points/181
    xy_points = np.zeros([2, num_points])

    for i in range(num_points):
        # throw out points farther than 5m
        if distances[i] <= 5:
            xy_points[:,i] = [distances[i]*np.cos(np.deg2rad(i/angle_scale_factor)),
                              distances[i]*np.sin(np.deg2rad(i/angle_scale_factor))]
    return xy_points

# Generate Hough Transform Histogram
def generateHoughTransformHistogram(xy_points, xbins=100, ybins=100):
    num_points = xy_points.shape[1]

    x_vec = np.linspace(0, np.pi, xbins)
    y_vec = np.linspace(-5, 5, ybins)
    count_histogram = np.zeros([ybins, xbins])
    points_histogram = x = [[[]] * xbins for i in range(ybins)]
    theta = np.linspace(0, np.pi, 100)

    for i in range(num_points):
        # skip points at origin
        if xy_points[0][i] == 0 and xy_points[0][i] == 0:
            continue
        r = xy_points[0][i] * np.cos(theta) + xy_points[1][i] * np.sin(theta)
        for j in range(r.shape[0]):
            x_grid, = np.where(x_vec <= theta[j])
            y_grid, = np.where(y_vec <= r[j])
            try:
                x_grid = x_grid[-1]
                y_grid = y_grid[-1]
                count_histogram[y_grid][x_grid] += 1
                points_histogram[y_grid][x_grid] = points_histogram[y_grid][x_grid] + [i]
            except:
                pass
    return count_histogram, points_histogram

# Get Max Value In Hough Histogram
def getMax(points_histogram):
    max_x = -1
    max_y = -1
    max_count = 0
    for i in range(len(points_histogram)):
        for j in range(len(points_histogram[i])):
            if len(points_histogram[i][j]) > max_count:
                max_x = j
                max_y = i
                max_count = len(points_histogram[i][j])
    return max_x, max_y, max_count

# Hough Suppression
def supressPoints(voted_pts, points_histogram):
    for i in range(len(points_histogram)):
        for j in range(len(points_histogram[i])):
            points_histogram[i][j] = [x for x in points_histogram[i][j] if x not in voted_pts]
    return points_histogram

# Find Hough Peaks Corresponnding to Lines
def findHoughPeaks(points_histogram, count_histogram, threshold=20):
    max_x, max_y, max_count = getMax(points_histogram)
    lines = []
    point_assignments = []
    d_scale = interp1d([0,100],[-5,5])
    th_scale = interp1d([0,100],[0,np.pi])
    while max_count > threshold:
        lines += [[d_scale(max_y)+0, th_scale(max_x)+0]]
        voted_pts = points_histogram[max_y][max_x]
        point_assignments += [voted_pts]
        points_histogram = supressPoints(voted_pts, points_histogram)
        max_x, max_y, max_count = getMax(points_histogram)
    return lines, point_assignments

# Plot Lines Interactively
def plotLines(xy_points, lines):
    for line in lines:
        l = np.linspace(min(xy_points[0,:]), max(xy_points[0,:]))
        d = line[0]
        th = line[1]
        p = d * np.array([np.cos(th), np.sin(th)]).T
        v = np.array([np.sin(th), -np.cos(th)]).T
        p1 = p + v*10;
        p2 = p - v*10;
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
    plt.scatter(xy_points[0,:], xy_points[1,:])
    plt.xlim((-15,15))
    plt.show()

# Perform LS Line Fitting From Point Assignments
def linearLeastSquaresFit(point_assignments, distances, xy_points):
    num_points = distances.shape[0]
    angle_scale_factor = num_points/181
    lines = []

    # calculate u
    for point_set in point_assignments:
        u = np.zeros([2,1])
        for q in range(len(point_set)):
            temp = distances[point_set[q]]*np.array([[np.cos(np.deg2rad(point_set[q]/angle_scale_factor)),
             np.sin(np.deg2rad(point_set[q]/angle_scale_factor))]]).T
            u = u + temp
        u = u/len(point_set)
        # print u
        # calculate A
        A = np.zeros([2,2])
        for q in range(len(point_set)):
            dif = (u.T - xy_points[:, point_set[q]]).T
            res = np.matmul(dif, dif.T)
            A = A + res
        # calculate parameters
        w, v = np.linalg.eig(A)
        eigvec = v[:,np.argmin(w)]
        th = np.arctan2(eigvec[1], eigvec[0])
        d = np.matmul(u.T, np.array([[np.cos(th), np.sin(th)]]).T)[0][0]
        lines += [[d, th]]

    return lines

# Find Lines From Laser Data
def findLines(distances):
    xy_points = getLocalCoordinatesFromLaserData(distances)
    count_histogram, points_histogram = generateHoughTransformHistogram(xy_points)
    lines, point_assignments = findHoughPeaks(points_histogram, count_histogram)
    lines = linearLeastSquaresFit(point_assignments, distances, xy_points)
    return xy_points, lines

# Throw Out Similar Parameter Lines
def throw_out_similar_lines(lines):
  num_lines = len(lines)
  lines_to_throw = []
  for i in range(num_lines):
    for j in range(i+1, num_lines):
      if abs(lines[i][0] - lines[j][0]) < 0.5 and abs(lines[i][1] - lines[j][1]) < 0.5:
        lines_to_throw.append(j)
      elif abs(lines[i][0] - lines[j][0]) < 0.5 and abs(lines[i][1]) >= 3 and abs(lines[j][1]) >= 3:
        lines_to_throw.append(j)
  new_lines = [lines[l] for l in range(len(lines)) if l not in lines_to_throw]
  return new_lines

# Convert Lines to Positive D Values
def standardizeLines(lines):
    for i in range(len(lines)):
        d = lines[i][0]
        th = lines[i][1]
        if d < 0:
            if th < 0:
                lines[i][1] += np.pi
            else:
                lines[i][1] -= np.pi
            lines[i][0] = abs(lines[i][0])
    return lines

'''
# USAGE:
import find_lines
distances = find_lines.loadData('scan_0')
xy_points = find_lines.getLocalCoordinatesFromLaserData(distances)
count_histogram, points_histogram = find_lines.generateHoughTransformHistogram(xy_points)
lines, point_assignments = find_lines.findHoughPeaks(points_histogram, count_histogram)
find_lines.plotLines(xy_points, lines)
lines = find_lines.linearLeastSquaresFit(point_assignments, distances, xy_points)
find_lines.plotLines(xy_points, lines)
'''