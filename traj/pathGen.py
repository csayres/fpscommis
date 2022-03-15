from kaiju.robotGrid import RobotGridCalib
from multiprocessing import Pool
import time
import pandas
import numpy
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy.signal import savgol_filter
from rdp import rdp
from simplification.cutil import simplify_coords


SPEED = 2
ANG_STEP = 0.1
SMOOTH_PTS_OLD = 5
COLLISION_SHRINK_OLD = 0.08
EPSILON_FACTOR_OLD = 2
EPSILON_OLD = ANG_STEP * EPSILON_FACTOR_OLD
COLLISION_BUFFER = 2
PATH_DELAY = 1

SMOOTH_PTS_NEW = 13
EPSILON_FACTOR_NEW = 2
EPSILON_NEW = ANG_STEP * EPSILON_FACTOR_NEW
COLLISION_SHRINK_NEW = 0.08

softLim = SPEED * 360 / 60

seeds1 = [93, 85, 79, 73, 72, 70, 62, 57, 55, 52, 46, 42, 33, 26, 22, 19, 11, 4, 1]
seeds2 = [
    122, 119, 118, 114, 110, 108, 105, 104, 103, 102, 195, 193, 192, 190, 183, 181, 179, 177, 176, 175,
    165, 162, 154, 148, 147, 146, 143, 140, 137, 133, 132, 131, 130, 129
]

seeds = seeds1 + seeds2

choiceSeeds = [93, 103, 73, 175, 154, 19, 140, 130] # some of the worst ones?


# def smoothPaths(seed, robotID=None):

#     subselect = "*"
#     if robotID is not None:
#         subselect = "%i"%robotID

#     pathFiles = glob.glob("rawPaths/raw-paths-%i-%s.csv"%(seed, subselect))

#     dfList = []
#     for ff in pathFiles:
#         dfList.append(pandas.read_csv(ff))

#     df = pandas.concat(dfList)

#     robotIDs = list(set(df.robotID))

#     for robotID in robotIDs:
#         _df = df[df.robotID == robotID]

#         # steps = _df.time.to_numpy()
#         angle = _df.beta.to_numpy()
#         # print("angle near end", angle[-3])
#         # if angle[-3] == 10:
#         #     continue
#         # pad
#         angle = numpy.hstack((numpy.zeros(SMOOTH_PTS) + angle[0], angle, numpy.zeros(SMOOTH_PTS) + angle[-1]))
#         steps = numpy.arange(len(angle))


#         # vel = numpy.gradient(angle)

#         tstart = time.time()
#         angleHat = savgol_filter(angle, SMOOTH_PTS, 3)
#         print("savgol took", time.time()-tstart)

#         tangle = numpy.array([steps, angleHat]).T

#         tstart = time.time()
#         # out = rdp(tangle, epsilon=EPSILON)
#         out = simplify_coords(tangle, EPSILON)
#         print("rdp points", len(out))
#         print("rdp took", time.time()-tstart)

#         # import pdb; pdb.set_trace


#         print("endpoint diffs", angle[0]-angleHat[0], angle[-1]-angleHat[-1])
#         print("endpoint diffs", angle[0]-out[0], angle[-1]-angleHat[-1])
#         # vHat = savgol_filter(vel, 7, 3)
#         fig, axs = plt.subplots(1,1, figsize=(10,8))
#         axs.plot(steps, angle, '.-k')
#         axs.plot(steps, angleHat, '.-r', alpha=0.5)
#         axs.plot(out[:,0], out[:,1], '--o', color="green", alpha=0.5)
#         # axs[0].set_xlim([-1,250])
#         # axs[1].plot(steps, vel, '.-k')
#         # axs[1].plot(steps, vHat, '.-r', alpha=0.5)
#         # axs[1].set_xlim([-10,250])
#         axs.set_title("robotID %i"%robotID)
#     plt.show()



    # robotIDs = list(set(df.robotID))

    # 115, 3 looks pretty pathalogical

    # for robotID in robotIDs:
    #     plt.figure(figsize=(13,8))
    #     plt.title("robotID %i"%robotID)
    #     _df = df[df.robotID==robotID]
    #     plt.plot(_df.time, _df.beta, '.-k', alpha=1)

    # plt.show()
    # plt.show()

    # sns.lineplot(x="time", y="beta", style="robotID", data=df)
    # plt.show()


def doOne(seed):
    rg = RobotGridCalib(
        stepSize=ANG_STEP,
        epsilon=EPSILON_OLD,
        seed=seed
    )

    rg.setCollisionBuffer(COLLISION_BUFFER)

    for robot in rg.robotDict.values():
        robot.setXYUniform()
        robot.setDestinationAlphaBeta(10, 170)

    rg.decollideGrid()

    tstart = time.time()
    rg.pathGenGreedy()

    # print("didFail", rg.didFail)
    if rg.didFail:
        return None
    # print("path gen took", time.time()-tstart)

    # dump all paths for further processing
    # for r in rg.robotDict.values():
    #     df = pandas.DataFrame({
    #         "time": [x[0] for x in r.alphaPath],
    #         "alpha": [x[1] for x in r.alphaPath],
    #         "beta": [x[1] for x in r.betaPath]
    #         })
    #     df["robotID"] = r.id
    #     df["seed"] = seed
    #     # print("npts", len(df))
    #     df.to_csv("raw-paths-%i-%i.csv"%(seed, r.id))


    # smoothPaths(rg)

    tstart = time.time()
    toDestOld, fromDestOld = rg._getPathPair(
        speed=SPEED, smoothPoints=SMOOTH_PTS_OLD,
        collisionShrink=COLLISION_SHRINK_OLD, pathDelay=PATH_DELAY
    )
    # print("path smoothing old took", time.time() - tstart)
    # print("smoothCollisions old", rg.smoothCollisions)

    tstart = time.time()
    toDestNew, fromDestNew = rg.getPathPair(
        speed=SPEED, smoothPoints=SMOOTH_PTS_NEW,
        collisionShrink=COLLISION_SHRINK_NEW, pathDelay=PATH_DELAY,
        epsilon=EPSILON_NEW
    )
    print("path smoothing new took", time.time() - tstart)
    print("smoothCollisions new", rg.smoothCollisions)

    # construct data frame from paths, including
    # velocity
    robotID = []
    axis = []
    t = []
    p = []
    v = []
    direction = []

    for d, _direction in zip([toDestOld, fromDestOld, toDestNew, fromDestNew], ["toDestOld", "fromDestOld", "toDestNew", "fromDestNew"]):
        for _robotID, _axisd in d.items():
            for _axis, traj in _axisd.items():
                tlast = None
                plast = None
                for _p, _t in traj:
                    robotID.append(_robotID)
                    axis.append(_axis)
                    t.append(_t)
                    p.append(_p)
                    direction.append(_direction)

                    if tlast is None:
                        v.append(0)
                    else:
                        dp = plast - _p
                        dt = tlast - _t
                        v.append(dp/dt)
                    tlast = _t
                    plast = _p

    df = pandas.DataFrame({
        "robotID": robotID,
        "axis": axis,
        "t": t,
        "p": p,
        "v": v,
        "direction": direction
    })

    df["seed"] = seed
    return df


def plotVelocities():
    df = pandas.read_csv("smoothed-simplified.csv")




    _df = df[(df.direction=="fromDestOld")]
    alpha = _df[_df.axis=="alpha"]
    beta = _df[_df.axis=="beta"]
    plt.figure(figsize=(13,8))
    plt.plot(alpha.t, alpha.v, '.', color="blue", alpha=0.3)
    plt.plot(beta.t, beta.v, '.', color="black", alpha=0.3)
    plt.axhline(softLim, color="red")
    plt.axhline(-softLim, color="red")
    plt.ylim([-220,220])


    _df = df[(df.direction=="toDestOld")]
    alpha = _df[_df.axis=="alpha"]
    beta = _df[_df.axis=="beta"]
    plt.figure(figsize=(13,8))
    plt.plot(alpha.t, alpha.v, '.', color="blue", alpha=0.3)
    plt.plot(beta.t, beta.v, '.', color="black", alpha=0.3)
    plt.axhline(softLim, color="red")
    plt.axhline(-softLim, color="red")
    plt.ylim([-220,220])


    _df = df[(df.direction=="fromDestNew")]
    alpha = _df[_df.axis=="alpha"]
    beta = _df[_df.axis=="beta"]
    plt.figure(figsize=(13,8))
    plt.plot(alpha.t, alpha.v, '.', color="blue", alpha=0.3)
    plt.plot(beta.t, beta.v, '.', color="black", alpha=0.3)
    plt.axhline(softLim, color="red")
    plt.axhline(-softLim, color="red")
    plt.ylim([-220,220])


    _df = df[(df.direction=="toDestNew")]
    alpha = _df[_df.axis=="alpha"]
    beta = _df[_df.axis=="beta"]
    plt.figure(figsize=(13,8))
    plt.plot(alpha.t, alpha.v, '.', color="blue", alpha=0.3)
    plt.plot(beta.t, beta.v, '.', color="black", alpha=0.3)
    plt.axhline(softLim, color="red")
    plt.axhline(-softLim, color="red")
    plt.ylim([-220,220])

    plt.show()


def generatePaths():
    tall = time.time()

    p = Pool(12)
    _dfList = p.map(doOne, seeds[:20])

    # _dfList = []
    # for seed in seeds[:5]:
    #     _dfList.append(doOne(seed))
    #     print("\n\n\n")
    # _dfList = [doOne(x) for x in seeds]

    # _dfList = [doOne(choiceSeeds[0])]

    print("all took", (time.time()-tall)/60)
    dfList = []
    for _df in _dfList:
        if _df is None:
            continue
        dfList.append(_df)

    print("%i succeeded"%len(dfList))

    df = pandas.concat(dfList)

    df.to_csv("smoothed-simplified.csv")

    # plt.figure(figsize=(13,8))
    # plotVelocities(df)
    # # plt.show()

    # for seed in list(set(df.seed)):
    #     plt.figure(figsize=(13,8))
    #     plt.title("seed %i"%seed)
    #     _df = df[df.seed==seed]
    #     plotVelocities(_df)

    # plt.show()


def oldVsNew():
    df = pandas.read_csv("smoothed-simplified.csv")
    df = df[(df.direction=="fromDestNew") | (df.direction=="fromDestOld")]

    robotIDs = list(set(df.robotID))

    for robotID in robotIDs:
        _df = df[(df.robotID==robotID)]

        plt.figure(figsize=(13,5))
        sns.lineplot(x="t", y="p", hue="axis", style="direction", data=_df)
        plt.title("robotID %s"%robotID)

        plt.show()





if __name__ == "__main__":

    # generatePaths()
    # oldVsNew()
    # plotVelocities()
    # generatePaths()
    # smoothPaths(93, 3)


    df = pandas.read_csv("smoothed-simplified.csv")
    robotIDs = list(set(df.robotID))

    _df = df[df.direction=="fromDestNew"]
    for robotID in robotIDs:
        for axis in ["alpha", "beta"]:
            print("tmax", numpy.max(_df.t))







