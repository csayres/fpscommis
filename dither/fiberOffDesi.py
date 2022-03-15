import glob
from astropy.io import fits
import pandas
import matplotlib.pyplot as plt
import numpy
import seaborn as sns

bf = "ditherBOSS*b1-analysis-b1.fits"
af = "ditherAPOGEE*-analysis-h.fits"


mjds = [59595, 59596, 59601, 59602]

_mjd = []
_field = []
_fiber = []
_xoff = []
_yoff = []
_xtel = []
_ytel = []
_fiberType = []
_expid = []



for fiberType, pp in zip(["ap", "boss"], [af, bf]):
    for mjd in mjds:
        desiFiles = glob.glob("/Users/csayres/fpscommis/dither/utah/%i/%s"%(mjd, pp))
        for desiFile in desiFiles:
            fieldID = int(desiFile.split("-")[1])
            desiFits = fits.open(desiFile)[1].data

            print("spectrograph=%s mjd=%i field=%i"%(fiberType, mjd, fieldID))

            # sumFile = desiFile.split("-analysis")[0] + ".fits"
            # sumfits = fits.open(sumFile)
            # find the corresponding summary file
            for row in desiFits:
                for ii in range(len(row["fiber"])):
                    _mjd.append(mjd)
                    _field.append(fieldID)
                    _fiber.append(row["fiber"][ii])
                    _xoff.append(row["xfiboff"][ii])
                    _yoff.append(row["yfiboff"][ii])
                    _xtel.append(row["xtel"][ii])
                    _ytel.append(row["ytel"][ii])
                    _expid.append(row["expid"][ii])
                    _fiberType.append(fiberType)

                # import pdb; pdb.set_trace()


df = pandas.DataFrame({
    "mjd": _mjd,
    "field": _field,
    "fiber": _fiber,
    "xoff": _xoff,
    "yoff": _yoff,
    "xtel": _xtel,
    "ytel": _ytel,
    "fiberType": _fiberType,
    "expid": _expid
})

df["rfiboff"] = numpy.sqrt(df.xoff**2 + df.yoff**2)
df["rtel"] = numpy.sqrt(df.xtel**2 + df.ytel**2)


plt.figure(figsize=(10,10))
sns.scatterplot(x="xtel", y="ytel", hue="fiberType", style="fiberType", data=df)
plt.axis('equal')
# plt.show()

plt.figure()
sns.histplot(data=df, x="rtel", hue="fiberType", element="step", stat="density", bins=numpy.linspace(0,30, 30))

plt.show()


apFibers = df[df.fiberType=="ap"]
bossFibers = df[df.fiberType=="boss"]

for _df in [apFibers, bossFibers]:
    fibers = list(set(_df.fiber))
    fiberType = str(set(_df.fiberType))
    for fiber in fibers:
        _df = df[df.fiber==fiber]
        plt.figure()
        plt.title("%s fiber %i"%(fiberType, fiber))
        plt.hist(_df.rfiboff, bins=20)
        plt.xlim([0,10])

        plt.show()
            # import pdb; pdb.set_trace()
        # fibers = ff.fiber[:,0]

        # import pdb; pdb.set_trace()
