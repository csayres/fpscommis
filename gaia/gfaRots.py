import numpy

# idx=[0,1,2,3,4,5]
# gfaID=[1,2,3,4,5,6]
xyWoks = [
    [-284.728, 163.148],
    [-0.387, 327.561],
    [283.560, 164.799],
    [284.338, -163.687],
    [0.861,  -327.986],
    [-283.768, -165.022]
]
zWok = 0
kHat = numpy.array([0,0,1])

xUnit = [0.4935, 1, 0.5053, -0.4998, -1, -0.5032]
# yUnit = [0.8701, 0.0044, ]

for ii in range(6):
    ix = xUnit[ii]
    iy = numpy.sqrt(1-ix**2)
    if ii in [2,3]:
        iy = -1*iy
    iHat = numpy.array([ix, iy, 0])
    jHat = numpy.cross(kHat,iHat)
    xWok,yWok = xyWoks[ii]
    a = tuple([
        ii, ii+1, xWok, yWok, iHat[0], iHat[1], iHat[2],
        jHat[0], jHat[1], jHat[2], kHat[0], kHat[1], kHat[2]
    ])
    print("%i,APO,%i,%.3f,%.3f,0,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"%a)

