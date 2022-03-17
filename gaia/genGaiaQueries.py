import numpy


raDecs = []

for ra in numpy.arange(0,360,15):
    raDecs.append([ra, 20])

raDecs.append([90, 42])
raDecs.append([105, 42])
raDecs.append([135, 42])
raDecs.append([330, 45])
raDecs.append([22.5*15, 60])
raDecs.append([10*15, 42])
raDecs.append([22*15, 20])
raDecs.append([22*15, 25])
raDecs.append([6*15, 25])

f = open("fullGaia.sql", "w")
f.write("\o gaiaFields.txt\n")

for ra, dec in raDecs:
    line = "SELECT solution_id, source_id, ra, dec, phot_g_mean_mag, parallax, pmra, pmdec from catalogdb.gaia_dr2_source WHERE phot_g_mean_mag < 18 AND q3c_radial_query(ra, dec, %.4f, %.4f, 2);\n"%(ra, dec)
    f.write(line)

f.close()
