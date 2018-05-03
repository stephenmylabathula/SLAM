from __future__ import division

from AriaPy import *
import numpy as np
import sys

from robot import Robot


''' INITIALIZE ARIA '''
chooseWiBox()
Aria.init()
argparser = ArArgumentParser(sys.argv)
argparser.loadDefaultArguments()
pioneer = ArRobot()
conn = ArRobotConnector(argparser, pioneer)
laserCon = ArLaserConnector(argparser, pioneer, conn)

if not conn.connectRobot(pioneer):
    print 'Error connecting to robot'
    Aria.logOptions()
    print 'Could not connect to robot, exiting.'
    Aria.exit(1)

sonar = ArSonarDevice()
pioneer.addRangeDevice(sonar)
pioneer.runAsync(True)

if not Aria_parseArgs():
    Aria.logOptions()
    Aria.exit(1)

print 'Connecting to laser ...'
laser = None
if laserCon.connectLasers():
    print 'Connected to lasers as configured in parameters'
    laser = pioneer.findLaser(1)
else:
    print 'Warning: unable to connect to lasers. Continuing anyway!'

# SETUP TELEOP
jdAct = ArActionJoydrive()
kdAct = ArActionKeydrive()
limiter = ArActionLimiterForwards("speed limiter near", 300, 600, 250)
limiterFar = ArActionLimiterForwards("speed limiter far", 300, 1100, 400)
tableLimiter = ArActionLimiterTableSensor()
backwardsLimiter = ArActionLimiterBackwards()
pioneer.lock()
pioneer.addAction(tableLimiter, 100)
pioneer.addAction(limiter, 95)
pioneer.addAction(limiterFar, 90)
pioneer.addAction(backwardsLimiter, 85)
pioneer.addAction(kdAct, 51)
pioneer.addAction(jdAct, 50)
pioneer.unlock()

# Enable Robot Motors
pioneer.enableMotors()

robot = Robot(pioneer)
get_scan = 5
while True:
    get_scan -= 1
    ArUtil.sleep(20)
    pioneer.lock()
    if laser and get_scan == 0:
        get_scan = 5
        # get laser data
        laser.lockDevice()
        laser_readings = laser.getRawReadingsAsVector()
        laser.unlockDevice()
        # get sonar data
        sonar_readings = np.abs([pioneer.getSonarReading(0).getLocalY(), pioneer.getSonarReading(3).getLocalY(),
                                 pioneer.getSonarReading(4).getLocalY(), pioneer.getSonarReading(7).getLocalY()])
        pioneer.unlock()
        robot.control(laser_readings, sonar_readings)
    else:
        pioneer.unlock()
