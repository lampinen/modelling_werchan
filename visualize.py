import numpy as np
import matplotlib.pyplot as plot

runs_to_plot = range(20, 40)
dataset_names = ["dataset_%s" % d for d in ["1", "2", "1A", "3"]]
nlayer = 4

for run in runs_to_plot:
    filename_prefix = "results/nlayer_%i_rseed_%i_" %(nlayer, run)

    for name in dataset_names: 
        filename = filename_prefix + name + "_gen_phase_beginning_outputs.csv"
        X = np.loadtxt(filename, delimiter=",")

        filename = filename[:-3] + "png"
        plot.figure()
        plot.imshow(X)
        plot.colorbar()
        plot.savefig(filename)
        plot.close()

