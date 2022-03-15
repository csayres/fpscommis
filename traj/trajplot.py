import pandas
import matplotlib.pyplot as plt
import seaborn as sns

maxV = 3  # rpm
maxV = maxV * 360 / 60  # deg per sec
softV = 2
softV = softV * 360 / 60  # deg per sec

badTrajIDs = ["trajectory-59598-0136", "trajectory-59598-0146"]

df = pandas.read_csv("trajBank.csv")

tids = list(set(df.trajID))

plt.figure(figsize=(13, 10))
allAlpha = df[df.axis == "alpha"]
allBeta = df[df.axis== "beta"]

plt.plot(df.time.to_numpy(), df.velocity.to_numpy(), '.k', alpha=0.5)

# plot failed trajectorys
fails = df[df.success==False]
plt.plot(fails.time.to_numpy(), fails.velocity.to_numpy(), 'xb', alpha=0.5)



# for tid in tids:
#     for color, axis in zip(["black", "blue"], ["alpha", "beta"]):
#         _df = df[(df.trajID == tid) & (df.axis == axis)]
#         plt.plot(_df.time, _df.velocity, color=color, alpha=0.5)

for tid, color in zip(badTrajIDs, ["orange", "purple"]):
    _df = df[df.trajID == tid]
    plt.plot(_df.time, _df.velocity, color=color, alpha=1)


plt.axhline(maxV, color="red")
plt.axhline(-maxV, color="red")

plt.axhline(softV, color="blue")
plt.axhline(-softV, color="blue")

plt.show()

#     alpha = _df[_df.axis="alpha"]


# totalTraj = 0
# overHardLim = []
# overSoftLim = []
# for mjd in mjds:
#     _df = df[df.mjd==mjd]
#     tids = list(set(_df.trajID))
#     totalTraj += len(tids)
#     for tid in tids:
#         _df2 = _df[_df.trajID==tid]
#         _maxV = _df2.maxVel.to_numpy()[0]
#         if _maxV > maxV:
#             overHardLim.append([mjd, tid])
#         elif _maxV > softV:
#             overSoftLim.append([mjd, tid])

# print("total trajectories", totalTraj)
# print("over max v", len(overHardLim))
# print("over soft v", len(overSoftLim))



# df = df.set_index(["mjd", "trajID"])
# import pdb; pdb.set_trace()
# count total mjd, traj id pairs
