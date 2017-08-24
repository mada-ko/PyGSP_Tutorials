import numpy as np
from pygsp import graphs, filters

G = graphs.Logo()
G.plot(default_qtg=False)
G.compute_fourier_basis()

G.plot_signal(G.U[:, 1], vertex_size=50, default_qtg=False)
G.plot_signal(G.U[:, 2], vertex_size=50, default_qtg=False)
