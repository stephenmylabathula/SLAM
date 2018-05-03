from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import hough
from slam_temp import SLAM_update


class Robot:
    def __init__(self, robot):
        self.robot = robot
        self.time_step = 100  # ms
        self.mode = 0
        self.robot.setDirectMotionPrecedenceTime(self.time_step)

        self.Q = np.array([[0.0075, 0.00075],
                           [0.00075, 0.0075]])
        self.Sig_msmt = np.array([[0.001, 0.0001],
                                  [0.0001, 0.001]])
        self.x_t = np.zeros((3, 1))
        self.Sig_t = 0.001 * np.eye(3)

        plt.ion()

    def SLAM_propagate(self):
        _x = self.x_t[0][0]
        _y = self.x_t[1][0]
        # _th = np.deg2rad(self.robot.getEncoderTh() * (90/55)) # For physical robot
        _th = self.x_t[2][0]

        delta_t = 1. / 2.15
        vel = self.robot.getVel() / 1000
        rot = np.deg2rad(self.robot.getRotVel())

        # for regular coordinates x -> sin, y -> cos
        self.x_t[0] = _x + vel * delta_t * np.cos(_th)
        self.x_t[1] = _y + vel * delta_t * np.sin(_th)
        self.x_t[2] = _th + rot * delta_t

        P_RR_old = self.Sig_t[0:3, 0:3]

        # for regular coordinates x -> cos, y -> sin
        phi_x = -vel * delta_t * np.sin(_th)
        phi_y = -vel * delta_t * np.cos(_th)
        phi_th = 1
        Phi = np.array([[1, 0, phi_x],
                        [0, 1, phi_y],
                        [0, 0, phi_th]])

        g_x = -delta_t * np.cos(_th)
        g_y = -delta_t * np.sin(_th)
        G = np.array([[g_x, 0],
                      [g_y, 0],
                      [0, -delta_t]])

        self.Sig_t[0:3, 0:3] = np.matmul(np.matmul(Phi, P_RR_old), Phi.T) + np.matmul(np.matmul(G, self.Q), G.T)

        for i in range(3, self.Sig_t.shape[1], 2):
            curr_submatrix = self.Sig_t[0:3, i:i + 2]
            self.Sig_t[0:3, i:i + 2] = np.matmul(Phi, curr_submatrix)

        for i in range(3, self.Sig_t.shape[0], 2):
            curr_submatrix = self.Sig_t[i:i + 2, 0:3]
            self.Sig_t[i:i + 2, 0:3] = np.matmul(curr_submatrix, Phi.T)

    # from MATLAB gets difference between two angles
    def angdiff(self, v1, v2):
        v1x = np.cos(v1)
        v1y = np.sin(v1)
        v2x = np.cos(v2)
        v2y = np.sin(v2)
        return np.arccos(np.matmul(np.array([v1x, v1y]), np.array([v2x, v2y]).T))

    def parallelism(self, v1, v2):
        return np.sin(self.angdiff(v1, v2))

    def findParallelWalls(self, distances, thetas):
        threshold = 0.1
        for i in range(len(thetas)):
            for j in range(i + 1, len(thetas)):
                if self.parallelism(thetas[i], thetas[j]) < threshold:
                    l1 = (distances[i], thetas[i])
                    l2 = (distances[j], thetas[j])
                    return [l1, l2] if abs(thetas[0]) > np.pi/2 else [l2, l1]
        return None

    def set_control_input(self, v, w):
        self.robot.setDirectMotionPrecedenceTime(self.time_step)
        self.robot.lock()
        self.robot.clearDirectMotion()
        self.robot.setVel(v)
        self.robot.setRotVel(w)
        self.robot.unlock()

    def control(self, laser_readings, sonar_readings):
        N = len(laser_readings)
        if N == 0:
            return
        d_theta = np.zeros((N, 2))

        scale = 15000

        # NOTE: laser reads from left to right as 0 to 180
        for i in range(N):
            dist_cand = laser_readings[i].getRange()
            theta_cand = laser_readings[i].getSensorTh()
            if dist_cand < scale:
                d_theta[N - i - 1][0] = dist_cand
                d_theta[N - i - 1][1] = theta_cand

        distances = d_theta[:, 0] / 1000
        xy_points = hough.getLocalCoordinatesFromLaserData(distances)
        count_histogram, points_histogram = hough.generateHoughTransformHistogram(xy_points)
        lines, point_assignments = hough.findHoughPeaks(points_histogram, count_histogram)
        lines = hough.linearLeastSquaresFit(point_assignments, distances, xy_points)

        lines = hough.throw_out_similar_lines(lines)
        lines = hough.standardizeLines(lines)
        hough.plotLines(xy_points, lines)

        distances = []
        thetas = []
        landmarks = np.array([[], []])
        for line in lines:
            distances.append(line[0])
            thetas.append(line[1])
            landmarks = np.concatenate((landmarks, np.array([[line[0]], [line[1]]])), 1)

        self.drive(distances, thetas, laser_readings, sonar_readings)
    
        self.SLAM_propagate()
        print self.x_t[0:3], " | Uncertainty: ", np.linalg.det(self.Sig_t[0:3, 0:3])
        if landmarks.shape[1] > 0:
            self.x_t, self.Sig_t = SLAM_update(self.x_t, self.Sig_t, landmarks, self.Sig_msmt)

        plt.pause(0.01)
        plt.clf()

    def drive(self, distances, thetas, laser_readings, sonar):
        walls = self.findParallelWalls(distances, thetas)
        #print walls
        raw_laser = [i.getRange() for i in laser_readings]
        laser_map = [np.mean(raw_laser[0:73]), np.mean(raw_laser[73:145]),
                     np.mean(raw_laser[145:217]), np.mean(raw_laser[217:289]),
                     np.mean(raw_laser[289:361])]
        # print(laser_map)
        if self.mode == 0:
            # print "HALLWAY"
            if laser_map[0] > 5000 and laser_map[2] < 2000:
                # open to the left close to wall in front
                self.mode = 2
            elif sonar[1] <= 100 or sonar[2] <= 100 or laser_map[2] < 1000:
                # very close to obstacle in front
                self.mode = 1
            elif sonar[0] < 100:
                # very close to left wall
                # turn right while going forward
                self.set_control_input(500, -10)
            elif sonar[3] < 100:
                # very close to right wall
                # turn left while going forward
                self.set_control_input(500, 10)
            elif not walls:
                # go straight
                self.set_control_input(100, 0)
            elif abs(walls[0][0]) < 1:
                # close to left wall
                # turn right strafe
                self.set_control_input(500, -5)
            elif abs(walls[1][0]) < 1:
                # close to right wall
                # turn left strafe
                self.set_control_input(500, 5)
            elif abs(walls[0][1]) < 3:
                if walls[0][1] > 0:
                    # turn right
                    self.set_control_input(0, -5)
                else:
                    # turn left
                    self.set_control_input(500, 5)
            else:
                # go straight
                self.set_control_input(500, 0)

        elif self.mode == 1:
            print "CORNER_TURN"
            walls = self.findParallelWalls(distances, thetas)
            if walls and -0.2 < walls[1][1] < 0.2:
                self.mode = 0
            else:
                self.set_control_input(0, 10)

        elif self.mode == 2:
            print "CORNER_STRAFE"
            walls = self.findParallelWalls(distances, thetas)
            if walls and -0.2 < walls[1][1] < 0.2:
                self.mode = 0
            else:
                self.set_control_input(100, 10)
