import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns


def dataGen(camera):
    # df = pandas.read_csv("dither-boss-59596-None-None.csv")
    # df2 = pandas.read_csv("dither-boss-59595-None-None.csv")
    # df = pandas.concat([df, df2])

    df = pandas.read_csv("ap-boss-dithers-latest.csv")

    import pdb; pdb.set_trace()

    df = df[df.camera==camera]

    df["MJD"] = df.MJD_x


    print(len(set(df.design_id)), "designs")
    print(set(df.MJD), "mjds")
    # df["configuration_id"] = df.configuration_id


    designs = list(set(df.design_id))
    mjds = list(set(df.MJD))
    fiberIDs = list(set(df.fiber))


    _fiberID = []
    _designID = []
    _xWokModel = []
    _yWokModel = []
    _xWokMeas = []
    _yWokMeas = []
    _alpha = []
    _beta = []
    _design = []
    _mjd = []

    for fiberID in fiberIDs:
        for mjd in mjds:
            for design in designs:
                # print("on", fiberID, mjd, design)
                _df = df[(df.fiber==fiberID) & (df.design_id==design) & (df.MJD==mjd)]

                if len(_df) == 0:
                    continue

                parent = list(set(_df.parent_configuration) - set([-999]))
                if len(parent) != 1:
                    # print("skipping design", design, "no parent")
                    continue

                if True in _df.isSky.to_numpy():
                    # print("skippping sky fiber")
                    continue

                parent = parent[0]

                _pdf = _df[(_df.configuration_id == parent) & (_df.fvc == False)]

                # print("len pdf", len(_pdf))
                _pdf = _pdf.nlargest(1, "expid")
                # import pdb; pdb.set_trace()

                if False in _pdf.activeFiber.to_numpy():
                    # print("skipping inactive fiber")
                    continue

                xWokModel = _pdf.xwok.to_numpy()[0]
                yWokModel = _pdf.ywok.to_numpy()[0]

                # remove the non-fvc parent config from the dataframe
                _df = _df[~((_df.configuration_id == parent) & (_df.fvc == False))]
                # _dithered = _df[_df.parent_configuration==parent]
                # import pdb; pdb.set_trace()



                # pick 3 brightest dithers and average
                _top = _df.nlargest(5, columns=["spectroflux"])
                # remove negative fluxes
                _top = _top[_top.spectroflux > 0]
                xwoks = _top.xwok.to_numpy()
                ywoks = _top.ywok.to_numpy()

                fluxes = _top.spectroflux.to_numpy()
                fluxnorm = fluxes / numpy.sum(fluxes)
                xWokMeas = numpy.sum(xwoks*fluxnorm)
                yWokMeas = numpy.sum(ywoks*fluxnorm)

                # xWokMeas = numpy.mean(xwoks)
                # yWokMeas = numpy.mean(ywoks)

                alpha = float(_pdf.alpha)
                beta = float(_pdf.beta)

                _fiberID.append(fiberID)
                _designID.append(design)
                _xWokModel.append(xWokModel)
                _yWokModel.append(yWokModel)
                _xWokMeas.append(xWokMeas)
                _yWokMeas.append(yWokMeas)
                _alpha.append(alpha)
                _beta.append(beta)
                _design.append(design)
                _mjd.append(mjd)



            # plt.figure()
            # sns.scatterplot(x="xwok", y="ywok", hue="spectroflux", size="spectroflux", data=_df)
            # plt.plot(xWokModel, yWokModel, "xr")
            # plt.plot(xWokMeas, yWokMeas, "o", markerfacecolor="none", markeredgecolor="red")
            # # plt.show()

            # plt.figure()
            # plt.hist(_df.alpha)
            # plt.xlabel("alpha")

            # plt.figure()
            # plt.hist(_df.beta)
            # plt.xlabel("beta")

            # plt.show()

            # import pdb; pdb.set_trace()
            # print("xWokModel", xWokModel)
            # import pdb; pdb.set_trace()

    fiberDF = pandas.DataFrame({
        "fiberID": _fiberID,
        "designID": _designID,
        "xWokModel": _xWokModel,
        "yWokModel": _yWokModel,
        "xWokMeas": _xWokMeas,
        "yWokMeas": _yWokMeas,
        "alpha": _alpha,
        "beta": _beta,
        "design": _design,
        "mjd": _mjd
    })
    fiberDF.to_csv("fiberDF.csv")

dataGen(camera="b1")

df = pandas.read_csv("fiberDF.csv")
df["dx"] = df.xWokModel - df.xWokMeas
df["dy"] = df.yWokModel - df.yWokMeas

# rotate into beta frame
# 86:         rotMat = numpy.array([
#   287              [cos, sin],
#   288              [-sin, cos]
#   289          ])
df["rotAng"] = -90 + df.alpha + df.beta


c = numpy.cos(numpy.radians(df.rotAng))
s = numpy.sin(numpy.radians(df.rotAng))

# plt.figure()
# plt.hist(c)
# plt.show()

# plt.figure()
# plt.hist(s)
# plt.show()

# import pdb; pdb.set_trace()

df["bx"] = df.dx * c + df.dy * s
df["by"] = -df.dx * s + df.dy * c


fibers = list(set(df.fiberID))


for fiber in fibers:
    _df = df[df.fiberID==fiber]
    print("rot", _df.rotAng.to_numpy())
    print("nObs", len(_df.dx))
    print("designs", set(_df.design))
    print("mjds", set(_df.mjd))
    plt.figure(figsize=(10,10))
    plt.plot(_df.dx, _df.dy, '.k', markersize=10)
    plt.plot(_df.bx, _df.by, '.r', markersize=10)

    # for x,y,mjd,design in zip(_df.bx, _df.by, _df.mjd, _df.design):
    #     plt.text(x,y,"%i-%i"%(mjd,design))
    plt.grid("on")
    plt.xlim([-.3, .3])
    plt.ylim([-.3, .3])
    plt.title("fiber %i"%fiber)
    plt.show()

# import pdb; pdb.set_trace()