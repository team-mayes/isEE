import numpy as np
import pytraj

muts = ['WT', '196LYS_733LYS_1270LYS_1807LYS','192SER_729SER_1266SER_1803SER','192ALA_729ALA_1266ALA_1803ALA','78PRO_615PRO_1152PRO_1689PRO','78GLU_196LYS_615GLU_733LYS_1152GLU_1270LYS_1689GLU_1807LYS','78SER_196LYS_615SER_733LYS_1152SER_1270LYS_1689SER_1807LYS','78ARG_196LYS_280GLY_615ARG_733LYS_817GLY_1152ARG_1270LYS_1354GLY_1689ARG_1807LYS_1891GLY','78MET_196LYS_280GLY_427GLY_615MET_733LYS_817GLY_964GLY_1152MET_1270LYS_1354GLY_1501GLY_1689MET_1807LYS_1891GLY_2038GLY']
scores = [1, 1.99, 1.6, 1.09, 2.02, 2.51, 2.02, 2.32, 1.23]
mask = ':1899,1902,1217,2149'
rmsfs = []  # lists of RMSF values for individual atoms in mask

# Get rmsfs
for mut in muts:
    traj = pytraj.iterload('working_directory_oct_restrained/pal_ts_MDO_oct.inpcrd_' + mut + '.nc','working_directory_oct_restrained/pal_ts_MDO_oct.inpcrd_' + mut + '_tleap.prmtop')
    traj = traj[int(traj.n_frames / 10):]
    rmsfs.append([item[1] for item in list(pytraj.rmsf(traj, mask))])

# Do optimization
best = 0
for ii in range(10000):
    lower = np.random.randint(0,len(rmsfs[0]) - 1)
    upper = np.random.randint(lower, len(rmsfs[0]))
    this_rmsfs = [np.mean(rmsfs[jj][lower:upper]) for jj in range(len(rmsfs))]
    corr = np.corrcoef(this_rmsfs, scores)[0][0]
    if corr < best:
        best = corr
        best_indices = [lower, upper]

        print(best)
        print(best_indices)
