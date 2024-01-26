from scipy.spatial.distance import cdist
from .utils import make_color
from mayavi import mlab
import numpy as np
import networkx as nx



def plot_mol_3d(pos,
                idx_i,
                idx_j,
                filename,
                at_size = 0.5,
                bond_size = 0.07,
                edge_size = .1,
                edge_rel=None,
                # relevances,
                rel_scaling=1.,
                # random_scal,
                bgc=(1, 1, 1),
                mol_rot = (0,30),
                color_mode = 'rb'
                ):

    at_c = []
    at_opacities = []

    at_c = [(0.9, 0.9, 0.9) for _, _ in enumerate(pos)]  # white/grey
    at_opacities = [1.0 for _, _ in enumerate(pos)]

    # create empty figure
    mlab.figure(1, bgcolor=bgc, size=(1200, 1200))
    mlab.clf()

    # plot molecule
    for i, p in enumerate(pos):
        # plot atom as sphere (3d point)
        mlab.points3d([p[0]], [p[1]], [p[2]],
                      scale_factor=at_size,
                      resolution=20,
                      color=at_c[i],
                      scale_mode='none',
                      opacity=at_opacities[i])

    cmat = cdist(pos, pos)
    for i,j in zip(idx_i, idx_j):
        if cmat[i,j] >1.6: continue
        p = pos[[i, j]]
        mlab.plot3d(p[:, 0], p[:, 1], p[:, 2],
                    color=(0.3, 0.3, 0.3),
                    tube_radius=bond_size )

    if edge_rel is not None:
        for (i,j), rel in edge_rel.items():
            if i == j:
                p=pos[i]
                color = make_color(rel, scaling=rel_scaling)
                mlab.points3d([p[0]], [p[1]], [p[2]],
                              scale_factor=at_size,
                              resolution=20,
                              color=color,
                              scale_mode='none',
                              opacity=1)
            else:
                p = pos[[i, j]]
                color = make_color(rel, scaling=rel_scaling)
                mlab.plot3d(p[:, 0], p[:, 1], p[:, 2],
                            color=color,
                            tube_radius=edge_size,
                            opacity= min(1, abs(rel)*rel_scaling) )

    # place camera somehow
    (azimuth, elevation) = mol_rot
    mlab.view(azimuth=azimuth,
        elevation=elevation,
        distance=9,
        focalpoint='auto')
    # save figure
    if filename is not None:
        mlab.savefig(filename)
    # show figure
    mlab.close()

def vis_mol_2d(ax,
            atomic_numbers,
            pos,
           projdim=1,
           with_atom_id=False,
              verbose_out=False,
              with_atom_type=True):

    dists = cdist(pos,pos)
    graph = (dists<1.6).astype(float)
    graph -= np.eye(graph.shape[0])

    #project
    pos= pos[:,[d for d in [0,1,2] if d != projdim]] #{i:p[1:] for i, p in enumerate(pos)}

    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    if with_atom_id:
        names = [f"{atom_names_dict[Z.item()]}$^{i}$" for i, Z in enumerate(atomic_numbers)]
    else:
        names = [atom_names_dict[Z.item()] for i, Z in enumerate(atomic_numbers)]
    G = nx.from_numpy_array(graph)
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", node_size=4000, ax=ax)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=5,
        style="dotted",
        node_size=700,
        ax=ax
    )

    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    if with_atom_type:
        nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(names)},
                        font_weight='bold', font_size=40, ax=ax)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    if verbose_out:
        return ax, G, pos
    else:
        return ax
