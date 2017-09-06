import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
%matplotlib inline

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 5)

G = graphs.Logo()
G.compute_fourier_basis()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for i, ax in enumerate(axes):
    G.plot_signal(G.U[:, i], vertex_size=30, ax=ax)
    ax.set_title('Eigenvector {}'.format(i+1))
    ax.set_axis_off()

fig.tight_layout()

foo = np.arange(6)
foo = foo[:, np.newaxis]
