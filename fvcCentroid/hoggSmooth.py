import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import time

IMAX = 32 # maximum integer wave number
DELTAK = 2. * np.pi / 10000.0 # wave number spacing in inverse pixels
SAVE_COEFFS = False

# functions to set up design matrices


def fourier_functions(xs, ys):
    n = len(xs)
    assert len(ys) == n
    fxs = np.zeros((n, IMAX * 2 + 2))
    fys = np.zeros((n, IMAX * 2 + 2))
    iis = np.zeros(IMAX * 2 + 2).astype(int)
    for i in range(IMAX+1):
        fxs[:, i * 2] = np.cos(i * DELTAK * xs)
        fys[:, i * 2] = np.cos(i * DELTAK * ys)
        iis[i * 2] = i
        fxs[:, i * 2 + 1] = np.sin((i + 1) * DELTAK * xs)
        fys[:, i * 2 + 1] = np.sin((i + 1) * DELTAK * ys)
        iis[i * 2 + 1] = i + 1
    return fxs, fys, iis


def design_matrix(xs, ys):
    fxs, fys, iis = fourier_functions(xs, ys)
    n, p = fxs.shape
    Xbig = (fxs[:, :, None] * fys[:, None, :]).reshape((n, p * p))
    i2plusj2 = (iis[:, None] ** 2 + iis[None, :] ** 2).reshape(p * p)
    return Xbig[:, i2plusj2 <= IMAX ** 2]


if __name__ == "__main__":
    # dataOrig = Table.read("dxyPixels.csv", format='ascii.csv')
    dataOrig = Table.read("dxyPixels.csv", format="ascii.csv")

    xs = dataOrig["x"]
    ys = dataOrig["y"]
    dxs = dataOrig["dx"]
    dys = dataOrig["dy"]

    xm = xs - np.mean(xs)
    ym = ys - np.mean(ys)
    r = np.sqrt(xm**2+ym**2)
    print("max r", np.max(r)*120/1000.)

    t1 = time.time()
    X = design_matrix(xs, ys)
    print("design_matrix took", time.time()-t1)
    print("X.shape", X.shape)

    n, p = X.shape

    np.random.seed(42)
    rands = np.random.uniform(size=n)

    train = rands <= 0.8
    test = rands > 0.8
    print(np.sum(train), np.sum(test))

    t1 = time.time()
    beta_x, resids, rank, s = np.linalg.lstsq(X[train], dxs[train], rcond=None)
    if SAVE_COEFFS:
        with open("beta_x.npy", "wb") as f:
            np.save(f, beta_x)

    print("fit took", time.time()-t1)
    dxs_hat = X[test] @ beta_x
    print(rank, min(s), max(s))

    print("original dx (test set) RMS:", np.sqrt(np.mean(dxs[test] ** 2)))
    print("dx - dx_hat (test set) RMS:", np.sqrt(np.mean((dxs[test] - dxs_hat) ** 2)))
    print("dx - dx_hat (test set) MAD:", np.sqrt(np.median((dxs[test] - dxs_hat) ** 2)))

    t1 = time.time()
    beta_y, resids, rank, s = np.linalg.lstsq(X[train], dys[train], rcond=None)
    print("fit took", time.time()-t1)
    dys_hat = X[test] @ beta_y
    print(rank, min(s), max(s))

    if SAVE_COEFFS:
        with open("beta_y.npy", "wb") as f:
            np.save(f, beta_y)


    print("original dy (test set) RMS:", np.sqrt(np.mean(dys[test] ** 2)))
    print("dy - dy_hat (test set) RMS:", np.sqrt(np.mean((dys[test] - dys_hat) ** 2)))
    print("dy - dy_hat (test set) MAD:", np.sqrt(np.median((dys[test] - dys_hat) ** 2)))


    dxs_hat = X @ beta_x
    dys_hat = X @ beta_y

    resid_dx = dxs - dxs_hat
    resid_dy = dys - dys_hat

    # import pdb; pdb.set_trace()

    fit, ax = plt.subplots(1,1, figsize=(10,10))
    q = ax.quiver(xs, ys, dxs, dys, angles="xy", units="xy", width=2, scale=0.005)
    ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    ax.set_xlabel("x CCD (pix)")
    ax.set_ylabel("y CCD (pix)")
    plt.axis("equal")

    fit, ax = plt.subplots(1,1, figsize=(10,10))
    q = ax.quiver(xs, ys, resid_dx, resid_dy, angles="xy", units="xy", width=2, scale=0.005)
    ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    ax.set_xlabel("x CCD (pix)")
    ax.set_ylabel("y CCD (pix)")
    plt.title("fit")
    plt.axis("equal")


    ######### new stuff ###############

    dataNew = Table.read("movie7/dxyPixels-sep.csv", format="ascii.csv")

    xs = dataNew["x"]
    ys = dataNew["y"]
    dxs = dataNew["dx"]
    dys = dataNew["dy"]
    X = design_matrix(xs, ys)

    dxs_hat = X @ beta_x
    dys_hat = X @ beta_y

    resid_dx = dxs - dxs_hat
    resid_dy = dys - dys_hat

    # fit, ax = plt.subplots(1,1, figsize=(10,10))
    # q = ax.quiver(xs, ys, dxs, dys, angles="xy", units="xy", width=2, scale=0.005)
    # ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    # ax.set_xlabel("x CCD (pix)")
    # ax.set_ylabel("y CCD (pix)")
    # plt.title("new bg sub")
    # plt.axis("equal")

    # fit, ax = plt.subplots(1,1, figsize=(10,10))
    # q = ax.quiver(xs, ys, resid_dx, resid_dy, angles="xy", units="xy", width=2, scale=0.005)
    # ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    # ax.set_xlabel("x CCD (pix)")
    # ax.set_ylabel("y CCD (pix)")
    # plt.title("fit new bg sub")
    # plt.axis("equal")

    plt.figure(figsize=(8,8))
    rresid = np.sqrt(resid_dx**2+resid_dy**2)
    rresid = rresid * 120
    plt.hist(rresid, bins=np.arange(0,30,5))






    plt.show()
