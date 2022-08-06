import numpy
from hoggSmooth import design_matrix
import matplotlib.pyplot as plt
import time


with open("beta_x.npy", "rb") as f:
    beta_x = numpy.load(f)

with open("beta_y.npy", "rb") as f:
    beta_y = numpy.load(f)

xticks = numpy.arange(0,8000,10)
yticks = numpy.arange(0,6000,10)

xgrid, ygrid = numpy.meshgrid(xticks,yticks)
xgrid = xgrid.flatten()
ygrid = ygrid.flatten()


t1 = time.time()
X = design_matrix(xgrid,ygrid)
print('design took', time.time()-t1)

t1 = time.time()
dx = X @ beta_x
dy = X @ beta_y
print('math took', time.time()-t1)


#dx_img = dx.reshape((len(xticks), len(yticks)))

dy_img = dy.reshape((len(yticks), len(xticks)))
keepInds = numpy.abs(dy_img) > 0.75
dy_img[keepInds] = numpy.nan

dx_img = dx.reshape((len(yticks), len(xticks)))
keepInds = numpy.abs(dx_img) > 0.75
dx_img[keepInds] = numpy.nan

plt.figure(figsize=(10,10))
plt.imshow(dy_img, origin="lower")
plt.colorbar()
plt.title("dy pixels")
plt.xlabel("x pixels / 10")
plt.ylabel("y pixels / 10")


plt.figure(figsize=(10,10))
plt.imshow(dx_img, origin="lower")
plt.colorbar()
plt.title("dx pixels")
plt.xlabel("x pixels / 10")
plt.ylabel("y pixels / 10")
plt.show()