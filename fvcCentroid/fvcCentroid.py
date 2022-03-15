import glob
from astropy.io import fits
import sep
import numpy
import pandas
from coordio.transforms import arg_nearest_neighbor, transformFromMetData, xyWokFromPosAngles
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

#### fvc params #####
zb_polids = [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29]
# zb_polids = [0, 1, 2]
exposure_time = 5.
fbi_level = 6.
background_sigma = 3.5
# background_sigma = 20
centroid_min_npix = 100
max_rough_fit_distance = 20.
max_fiducial_fit_distance = 5.
max_final_fit_distance = 1.
reference_rotator_position = 135.4
centre_rotation = numpy.array([4115., 3092.])
target_rms = 5
target_delta_rms = 1
max_fvc_iterations = 5

fvcFiles = glob.glob("/Volumes/futa/apo/data/fcam/59605/proc*.fits")

_fullTable = calibration.positionerTable.merge(calibration.wokCoords, on="holeID").reset_index()
fiducialCoords = calibration.fiducialCoords.reset_index()

kernel = numpy.array([[1., 2., 3., 2., 1.],
                   [2., 3., 5., 3., 2.],
                   [3., 5., 8., 5., 3.],
                   [2., 3., 5., 3., 2.],
                   [1., 2., 3., 2., 1.]])

kernel = numpy.array([[0., 0., 1., 0., 0],
                   [0., 1., 2., 1., 0.],
                   [1., 2., 10., 2., 1.],
                   [0., 1., 2., 1., 0.],
                   [0, 0., 1., 0., 0]])


kernel = numpy.array([
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0.5, 1., 0.5, 0],
                   [0, 1., 8., 1., 0],
                   [0, 0.5, 1., 0.5, 0],
                   [0, 0, 0, 0, 0]
])


def extract(
    image_data,
    centroid_min_npix,
    background_sigma
    winpos_sigma=0.7,
    winpos_box_size=3

    ):
    image_data = numpy.array(image_data, dtype=numpy.float32)

    bkg = sep.Background(image_data)
    bkg_image = bkg.back()

    data_sub = image_data - bkg_image

    objects = sep.extract(
        data_sub,
        background_sigma,
        # filter_kernel=kernel,
        # filter_type="conv",
        err=bkg.globalrms,
    )

    objects = objects[objects["npix"] > centroid_min_npix]

    # create masks and re-extract
    maskArr = numpy.ones(data_sub.shape, dtype=bool)

    # import pdb; pdb.set_trace()
    for ii in range(len(objects)):
        _xm = objects["xcpeak"][ii]
        _ym = objects["ycpeak"][ii]
        for xind in [_xm-1, _xm, _xm+1]:
            for yind in [_ym-1, _ym, _ym+1]:
                maskArr[yind, xind] = False

    xNew, yNew, flags = sep.winpos(
        data_sub,
        objects["xcpeak"],
        objects["ycpeak"],
        sig=0.7,
        mask=maskArr
    )

    # plt.figure()
    # plt.imshow(data_sub, origin="lower")
    # plt.plot(objects["x"], objects["y"], 'xr')
    # plt.plot(xNew, yNew, 'xg')
    # plt.show()




    # replaceInds = objects["y2"] > 12
    # # print("replacing", sum(replaceInds))
    # objects["x"][replaceInds] = xNew[replaceInds]
    # objects["y"][replaceInds] = yNew[replaceInds]



    # import pdb; pdb.set_trace()

    objects = pandas.DataFrame(objects)

    objects["xWin"] = xNew
    objects["yWin"] = yNew

    # newxy = sep.winpos(data_sub, objects["x"], objects["y"], sig=0.1)

    objects["x"] = xNew
    objects["y"] = yNew

    # import pdb; pdb.set_trace()

    # reset objects in danger band.

    # replace = objects.y2 > 12
    # dx = objects.x - objects.xWin
    # dy = objects.y - objects.yWin

    # dx = dx - numpy.mean(dx)
    # dy = dy - numpy.mean(dy)

    # err = numpy.sqrt(dx**2+dy**2)


    # print("rms", numpy.sqrt(numpy.mean(err)))
    # plt.figure()
    # plt.hist(err)

    # plt.figure(figsize=(8,8))
    # plt.quiver(objects.x, objects.y, dx, dy, angles="xy", units="xy")
    # plt.axis("equal")
    # plt.show()

    # import pdb; pdb.set_trace()

    return objects


def dataGen():
    ftList = []
    zbCoeffs = []
    imgNums = []
    fitRMS = []
    rotAng = []
    for file in fvcFiles:

        ff = fits.open(file)
        IPA = ff[1].header["IPA"]
        LED = ff[1].header["LED1"]
        ROTPOS = ff[1].header["ROTPOS"]
        imgNum = int(file.split("-")[-1].split(".")[0])
        if LED < 5.9:
            ff.close()
            continue



        dRot = numpy.radians(IPA-reference_rotator_position)
        sinRot = numpy.sin(dRot)
        cosRot = numpy.cos(dRot)

        ################ centroids
        # centroids = fitsTableToPandas(ff[8].data)
        centroids = extract(ff[1].data)

        xyCCD = centroids[["x", "y"]].to_numpy()

        _xyRot = xyCCD - centre_rotation

        rotMat = numpy.array([
            [cosRot, -sinRot],
            [sinRot, cosRot]
        ])

        _xyRot = (rotMat @ _xyRot.T).T
        xyRot = _xyRot + centre_rotation

        # plt.figure(figsize=(8,8))
        # plt.plot(xyCCD[:,0], xyCCD[:,1], 'xr')
        # plt.plot(xyRot[:,0], xyRot[:,1], '.k')
        # plt.axis('equal')
        # plt.show()

        centroids["xCCD"] = centroids.x
        centroids["yCCD"] = centroids.y

        centroids["xRot"] = xyRot[:,0]
        centroids["yRot"] = xyRot[:,1]

        centroids["x"] = xyRot[:,0]
        centroids["y"] = xyRot[:,1]


        ##################### posangles ########
        posAngles = fitsTableToPandas(ff[7].data)
        fullTable = posAngles.merge(_fullTable, on="positionerID")
        fullTable = xyWokFromPosAngles(fullTable, "Metrology")
        fullTransform, fullTable = transformFromMetData(centroids, fullTable, fiducialCoords, file, zb_polids)
        fullTable["ipa"] = IPA
        fullTable["rotpos"] = ROTPOS
        fullTable["led"] = LED

        fullTable["imgnum"] = imgNum

        # plt.imshow(ff[1].data, origin="lower")
        # _pltcent = centroids[centroids.y2 > 10]
        # plt.plot(_pltcent.xCCD, _pltcent.yCCD, '.r')
        # plt.show()

        # import pdb; pdb.set_trace()


        zbCoeffs.append(fullTransform.coeffs)
        imgNums.append(imgNum)
        rotAng.append(ROTPOS)
        fitRMS.append(fullTransform.unbiasedRMS)
        ftList.append(fullTable)

        ff.close()


    ft = pandas.concat(ftList)
    ft.to_csv("centData.csv")

dataGen()

ft = pandas.read_csv("centData.csv")
ft["roundness"] = numpy.sqrt(ft.x2**2+ft.y2**2)



mean = ft.groupby(["rotpos", "positionerID"]).mean().reset_index()



plt.figure(figsize=(8,8))
sns.scatterplot(x="xCCD", y="yCCD", hue="xy", data=ft, palette="vlag")
plt.axis("equal")

plt.figure(figsize=(8,8))
sns.scatterplot(x="xCCD", y="yCCD", hue="y2", data=ft)
plt.axis("equal")

plt.figure(figsize=(8,8))
sns.scatterplot(x="xCCD", y="yCCD", hue="x2", data=ft)
plt.axis("equal")

# ax = plt.gca()
# for ii, row in mean.iterrows():
#     e = Ellipse(xy=(row['xCCD'], row['yCCD']),
#                 width=6*row['a'],
#                 height=6*row['b'],
#                 angle=row['theta'] * 180. / numpy.pi)
#     e.set_facecolor('none')
#     e.set_edgecolor('red')
#     ax.add_artist(e)


plt.show()


_ft = ft[ft.rotpos.isin([135.4, 225.4])]


_ft135 = _ft.loc[_ft.rotpos==135.4]

# import pdb; pdb.set_trace()

x = _ft135.xWokExpect.to_numpy()
y = _ft135.yWokExpect.to_numpy()

dx = _ft135.xWokMetMeas.to_numpy() - x
dy = _ft135.yWokMetMeas.to_numpy() - y
err = numpy.sqrt(dx**2+dy**2)*1000

plt.figure()
plt.quiver(x[err<500],y[err<500],dx[err<500],dy[err<500],angles="xy", units="xy", scale=1/200)
plt.axis("equal")



pos0 = mean.loc[mean.rotpos == 135.4]
pos1 = mean.loc[mean.rotpos == 225.4]

dx = pos0.xWokMetMeas.to_numpy() - pos1.xWokMetMeas.to_numpy()
dy = pos0.yWokMetMeas.to_numpy() - pos1.yWokMetMeas.to_numpy()
err = numpy.sqrt(dx**2+dy**2)*1000
x = pos0.xWokMetMeas.to_numpy()
y = pos0.yWokMetMeas.to_numpy()

plt.figure()
plt.quiver(x[err<500],y[err<500],dx[err<500],dy[err<500],angles="xy", units="xy", scale=1/200)
plt.axis("equal")

plt.figure()
plt.hist(err[err<500])

print("rms err", numpy.sqrt(numpy.mean(err)))

plt.show()

# import pdb; pdb.set_trace()

# sns.scatterplot(x="xWokMetMeas", y="yWokMetMeas", hue="rotpos", data=_ft)
# plt.axis("equal")
# plt.show()

    # import pdb; pdb.set_trace()

    # out =



    # import pdb; pdb.set_trace()


