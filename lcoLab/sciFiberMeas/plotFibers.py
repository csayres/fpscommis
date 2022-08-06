from coordio.defaults import calibration
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

df = pandas.read_csv("positionerTable.sciFiberMeas.csv")

_df = df[df.apX > 4]
import pdb; pdb.set_trace()

# plt.figure()


# df = calibration.positionerTable.reset_index()

plt.figure()
plt.plot(df.metX, df.metY, '.k')
plt.plot(df.apX, df.apY, '.r')
plt.plot(df.bossX, df.bossY, '.b')
plt.axis("equal")
plt.show()