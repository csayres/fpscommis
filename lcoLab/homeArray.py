from kaiju.utils import plotOne, plotPaths
from kaiju.robotGrid import RobotGridCalib
import matplotlib.pyplot as plt

angStep = 0.1         # degrees per step in kaiju's rough path
epsilon = angStep * 2   # max error (deg) allowed in kaiju's path simplification
collisionBuffer = 3.2    # effective *radius* of beta arm in mm effective beta arm width is 2*collisionBuffer
seed = 1
alpha = 118
finalBeta = -10
makeMovie = False


# robot 620, 1010 # stuck at 180,180

rg = RobotGridCalib(angStep, epsilon, seed)
rg.setCollisionBuffer(collisionBuffer)

for robot in rg.robotDict.values():
    robot.setAlphaBeta(alpha, finalBeta)
    robot.setDestinationAlphaBeta(10,170)


# disabled robot list
rg.robotDict[50].setAlphaBeta(alpha, 180)
rg.robotDict[50].setDestinationAlphaBeta(alpha, 180)
rg.robotDict[50].isOffline = True

# rg.robotDict[1010].setAlphaBeta(0,180)
# rg.robotDict[1010].setDestinationAlphaBeta(0,180)
# rg.robotDict[1010].isOffline = True


ax = plotOne(1, rg, "test1.png", highlightRobot=[620,1010], isSequence=False)

# find collided robots
notHomed = []
for r in rg.robotDict.values():
    if r.isOffline:
        continue
    if rg.isCollided(r.id):
        print(r.id, "is collided")
        r.setAlphaBeta(alpha, 180)
        notHomed.append(r.id)

ax = plotOne(1, rg, "test2.png", highlightRobot=[620,1010], isSequence=False)


rg.pathGenGreedy()
print("path 1 fail", rg.didFail)
if makeMovie:
    plotPaths(rg, downsample=50, filename="home1.mp4")

nextAlpha = 58
print("nextAlpha", nextAlpha)

for r in rg.robotDict.values():
    if r.isOffline:
        continue
    if r.id in notHomed:
        r.setAlphaBeta(nextAlpha, finalBeta)
    else:
        r.setAlphaBeta(nextAlpha, 180)


ax = plotOne(1, rg, "test3.png", highlightRobot=[620,1010], isSequence=False)

rg.pathGenGreedy()
print("path 2 fail", rg.didFail)

if makeMovie:
    plotPaths(rg, downsample=50, filename="home2.mp4")
