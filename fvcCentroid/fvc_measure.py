#!/usr/bin/env python

import fitsio
import os
import sys
import argparse
import numpy as np
import sdss_access.path
import dimage.measure as dm

sdss_path = sdss_access.path.Path(release='sdss5', preserve_envvars=True)

simple_dtype = [('x', np.float32),
                ('y', np.float32),
                ('f', np.float32)]

# Thanks for this!  If either of you (or anyone else) wants to experiment with a set of frames I’ve been using for analysis, you could look at MJD=59661 image numbers 26-619.  This is a set of two configurations, imaged 11 times at rotator angles spaced by 15 degrees.  Theoretically the robots should not be moving at all (just the rotator is moving) within the same configuration set.  The headers “CONFIGID” and “ROTPOS” should let you group them into sets of 11.  The FVC scale is 120 microns / pixel.  

def fcamname(mjd=None, frame=None):
    imfile = os.path.join(os.getenv('FCAM_DATA_N'),
                          str(mjd),
                          'proc-fimg-fvc1n-{f:04}.fits')
    imfile = imfile.format(f=frame)
    return(imfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description='Pretty plot of the fields')

    parser.add_argument('-m', '--mjd', dest='mjd',
                        default=59661,
                        type=int, help='MJD')

    parser.add_argument('-s', '--start', dest='start',
                        default=26,
                        type=int, help='start frame number')

    parser.add_argument('-e', '--end', dest='end',
                        default=619,
                        type=int, help='start frame number')

    args = parser.parse_args()
    mjd = args.mjd
    fstart = args.start
    fend = args.end

    nframes = fend - fstart + 1
    for i in range(nframes):
        frame = fstart + i
        print("Frame {f}".format(f=frame))
        im, hdr = fitsio.read(fcamname(mjd=mjd, frame=frame), header=True)
        cen = fitsio.read(fcamname(mjd=mjd, frame=frame), ext='CENTROIDS')
        configid = np.int32(hdr['CONFIGID'])
        rotpos = np.float32(hdr['ROTPOS'])

        imbias = np.median(im, axis=0)
        imbias = np.outer(np.ones(im.shape[0],
                                  dtype=np.float32),
                          imbias)
        im = im - imbias

        off = 1022
        imtrim = im[:, off:-off]
        x, y, f = dm.simplexy(imtrim, psf_sigma=1.5, plim=500.,
                              maxper=1)
        x = x + off
        print(' - found {n} peaks'.format(n=len(x)))
        print(' - official found {n} peaks'.format(n=len(cen)))
        simple = np.zeros(len(x), dtype=simple_dtype)
        simple['x'] = x
        simple['y'] = y
        simple['f'] = f

        offset_dtype = np.dtype([('isimple', int),
                                 ('icentroids', int),
                                 ('xdiff', np.float32),
                                 ('ydiff', np.float32),
                                 ('xorig', np.float32),
                                 ('yorig', np.float32)])
        offsets = np.zeros(len(simple), dtype=offset_dtype)
        for isimple in np.arange(len(simple)):
            icentroids = np.where((np.abs(cen['xWinpos'] - simple['x'][isimple]) < 2.) &
                                  (np.abs(cen['yWinpos'] - simple['y'][isimple]) < 2.))[0]
            if(len(icentroids) == 0):
                icentroids = - 1
                xdiff = 0.
                ydiff = 0.
                xorig = 0.
                yorig = 0.
            else:
                icentroids = icentroids[0]
                xdiff = simple['x'][isimple] - cen['xWinpos'][icentroids]
                ydiff = simple['y'][isimple] - cen['yWinpos'][icentroids]
                xorig = simple['x'][isimple] - cen['x'][icentroids]
                yorig = simple['y'][isimple] - cen['y'][icentroids]
            offsets['isimple'][isimple] = isimple
            offsets['icentroids'][isimple] = icentroids
            offsets['xdiff'][isimple] = xdiff
            offsets['ydiff'][isimple] = ydiff
            offsets['xorig'][isimple] = xorig
            offsets['yorig'][isimple] = yorig

        print(' - std offsets ({x}, {y})'.format(x=offsets['xdiff'].std(),
                                                 y=offsets['ydiff'].std()))
        print(' - std offsets v orig ({x}, {y})'.format(x=offsets['xorig'].std(),
                                                        y=offsets['yorig'].std()))

        outfile = 'fcam-measure-{f}.fits'.format(f=frame)
        fitsio.write(outfile, im, header=hdr, clobber=True)
        fitsio.write(outfile, cen, extname='CENTROIDS',
                     clobber=False)
        fitsio.write(outfile, simple, extname='SIMPLE',
                     clobber=False)
        fitsio.write(outfile, offsets, extname='OFFSETS',
                     clobber=False)
        
