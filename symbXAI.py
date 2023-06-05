import torch
from functools import reduce
# import schnetpack as spk
import numpy as np


class Node:
    """ Helping class for the explainer functions"""
    def __init__(self, node_rep,lamb, parent , R, domain_restrict=None):
        self.node_rep = node_rep
#         self.node_id = node_id
        self.parent = parent
        self.R = R
        self.lamb = lamb
        self.domain_restrict = domain_restrict

    def neighbors(self):
        neighs = list(self.lamb[self.node_rep].nonzero().T[0].numpy())
        if self.domain_restrict is not None:
            neighs = [n for n in neighs if n in self.domain_restrict]
        return neighs

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f"node representation: {self.node_rep}; parent : {self.parent}"

    def is_root(self):
        return self.parent is None

    def get_walk(self):
        curr_node = self
        walk = [self.node_rep]
        while curr_node.parent is not None:
            curr_node = curr_node.parent
            walk.append(curr_node.node_rep)
        return tuple(walk)

class SymbXAI:
    def __init__(self,
                 layers,
                 x,
                 num_nodes,
                 lamb,
                 R_T=None,
                 batch_dim=False,
                 scal_val=1.
                 ):
        '''
        Init function. It basically sets some hyperparametes and saves the activations.
        '''
        # Save some parameter
        self.layers = layers
        self.num_layer = len(layers)
        self.node_domain = list(range(num_nodes))
        self.num_nodes = num_nodes
        self.batch_dim = batch_dim
        self.scal_val = scal_val

        # Some placeholder parameter for later
        self.walk_rels_tens = None
        self.walk_rels_list = None
        self.node2idn = None
        self.walk_rel_domain = None
        self.walk_rels_computed = False

        # Set up activations
        self.xs = [x.data]
        for layer in layers: x = layer(x); self.xs.append(x.data)

        # Set up for each layer the node range
        # self.node2idn = [{node: i for i, node in enumerate(self.node_domain)} for _ in range(self.num_layer)] + [{0: 0}]

        # Set up for each layer the dependencies
        self.lamb_per_layer = [lamb for _ in range(self.num_layer-1)] + [torch.ones(self.num_nodes).unsqueeze(0)]

        # Initialize the relevance
        if R_T is None:
            self.R_T = self.xs[-1].data.detach()
        else:
            self.R_T = R_T

    def _relprop_standard(self, act, layer, R, node_rep):
        '''
        Just a vanilla relevance propagation strategy per layer guided along the specified nodes.
        Todo: Does it work for shifted-softplus activation as well?
        '''
        # import pdb; pdb.set_trace()
        # Make act gradable
        act = act.data.requires_grad_(True)

        # Forward layer guided at the node representation.
        z = layer(act)[node_rep] if not self.batch_dim else layer(act)[0,node_rep]

        assert z.shape == R.shape, f'z.shape {z.shape}, R.shape {R.shape}'

        s = R/z
        (z*s.data).sum().backward(retain_graph=True)
        c = torch.nan_to_num(act.grad)
        R = act*c

        return R.data


    def _update_tree_nodes(self, act_id, R, node_rep, parent, domain_restrict=None):
        '''
        Update the nodes in the internal dependency tree, by the given hyperparameter.
        '''
        lamb = self.lamb_per_layer[act_id-1]
#         import pdb; pdb.set_trace()
        self.tree_nodes_per_act[act_id] += [ Node(node_rep,
                                                     lamb,
                                                     parent,
                                                     R[node_rep] if not self.batch_dim else R[0,node_rep] ,
                                                     domain_restrict=domain_restrict) ]

    def setup_walk_relevance_scores(self, domain_restrict=None, verbose=False):
        '''
        To setup the relevance scores in the quality of walks. All scores will be safed at self.walk_rels_tens and self.walk_rels_list.
        '''

        # Create data structure:
        self.tree_nodes_per_act = [[] for _ in range((self.num_layer + 1))]

        # Initialize the last relevance activation
        self._update_tree_nodes(self.num_layer,
                self.R_T,
                0 ,
                None,
                domain_restrict=domain_restrict)

        for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
            # Iterate over the nodes
            # import pdb; pdb.set_trace()
            for node in self.tree_nodes_per_act[layer_id+1]:
                # Compute the relevance
                R = self._relprop_standard(act, layer, node.R, node.node_rep)
                # Distribute the relevance to the neighbors
                for neigh_rep in node.neighbors():
                    self._update_tree_nodes(layer_id,
                                            R,
                                            neigh_rep,
                                            node,
                                            domain_restrict=domain_restrict)

        # save a few parameter
        if domain_restrict is None:
            self.walk_rel_domain = self.node_domain
        else:
            self.walk_rel_domain = domain_restrict
        self.node2idn = {node: i for i, node in enumerate(self.walk_rel_domain)}
        self.walk_rels_computed = True

        # Free memory
#         del self.tree_nodes_per_act
    def node_relevance(self, mode='DeltaR'):

        if mode == 'node@input':
            # Initialize the last relevance
            curr_node = Node(
                            0,
                            self.lamb_per_layer[self.num_layer -1],
                            None,
                            self.R_T[0] if not self.batch_dim else self.R_T[0,0],
                            domain_restrict=None
                            )


            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes
                R = self._relprop_standard(act,
                                        layer,
                                        curr_node.R,
                                        curr_node.node_rep)

                # Create new nodes
                new_node = Node(self.node_domain,
                                self.lamb_per_layer[layer_id -1],
                                curr_node,
                                R[self.node_domain] if not self.batch_dim else R[0,self.node_domain],
                                domain_restrict=None)

                curr_node = new_node
            node_rel = curr_node.R.sum(-1)*self.scal_val
        elif mode == 'DeltaR':
            node_rel = np.array([self.subgraph_relevance([node]) for node in self.node_domain ])

        return node_rel

    def edge_relevance(self,mode='DeltaR',
                        cust_edges=None,
                        with_selfloop=True,
                        from_walks=False):
        assert mode in  ['edge_in_walk','walk_in_edge','edge@input', 'DeltaR'], f'mode "{mode}" is not defined'

        if mode == 'edge@input':
            # Initialize the last relevance
            curr_node = Node(
                            0,
                            self.lamb_per_layer[self.num_layer -1],
                            None,
                            self.R_T[0] if not self.batch_dim else self.R_T[0,0],
                            domain_restrict=None
                            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                if layer_id >0:
                    # Simple first order backpropagation:
                    R = self._relprop_standard(act,
                                            layer,
                                            curr_node.R,
                                            curr_node.node_rep)

                    # Create new nodes
                    new_node = Node(self.node_domain,
                                    self.lamb_per_layer[layer_id -1],
                                    curr_node,
                                    R[self.node_domain] if not self.batch_dim else R[0,self.node_domain],
                                    domain_restrict=None )

                    curr_node = new_node
                else:
                    if cust_edges is None:
                        lamb = self.lamb_per_layer[layer_id]
                    else:
                        lamb = torch.zeros(self.lamb_per_layer[layer_id].shape)
                        for i,j in cust_edges: lamb[i,j] = 1

                    temp_node_domain = set(lamb.nonzero().T.numpy()[0])
                    init_nodes =  [Node(node_rep,
                                         lamb,
                                         None,
                                         curr_node.R[node_rep] if not self.batch_dim else R[0,node_rep])
                                    for node_rep in temp_node_domain ]
                    out_nodes = []
                    # Iterate over the nodes
                    for node in init_nodes:
                        # Compute the relevance
                        R = self._relprop_standard(act, layer, node.R, node.node_rep)
                        # Distribute the relevance to the neighbors
                        # import pdb; pdb.set_trace()
                        for neigh_rep in node.neighbors():
                            out_nodes += [ Node(neigh_rep,
                                                 None,
                                                 node,
                                                 R[neigh_rep] if not self.batch_dim else R[0,neigh_rep] ) ]

            # extract the edge relevances
            edge_rel = {}

            for node in out_nodes:
                walk, rel = node.get_walk(), node.R.data.sum().item()
                edge_rel[walk] = rel * self.scal_val

        elif mode == 'edge_in_walk':
            # get edges
            edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]

            wrel = self.walk_relevance(verbose=False)
            edge_rel = {}
            for edge in edges:
                erel = 0.
                for w,rel in wrel:
                    for t in range(len(w)-1):
                        if w[t:t+2] == edge:
                            erel += rel
                            break
                edge_rel[edge] = erel.item()/(self.num_layer -1)

        elif mode == 'walk_in_edge':
            # get edges
            edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]

            edge_rel = {edge: self.subgraph_relevance(edge).item() for edge in edges}

        elif mode == 'DeltaR':
            if cust_edges is None:
                # get edges
                edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]
                if not with_selfloop:
                    edges = [(i,j) for (i,j) in edges if i != j]
            else:
                edges = cust_edges
            edge_rel = {}

            if from_walks:
                wrel = self.walk_relevance(verbose=False)
                for (i,j) in edges:
                    edge_rel[(i,j)] = 0.
                    for walk,rel in wrel:
                        if i != j and len(set(walk))==2 and all([w in [i,j] for w in walk]):
                            edge_rel[(i,j)] += float(rel)
                        elif i == j and len(set(walk))==1 and all([w in [i,j] for w in walk]):
                            edge_rel[(i,j)] += float(rel)
                        else:
                            continue
            else:
                for (i,j) in edges:
                    if i == j:
                        edge_rel[(i,j)] = self.subgraph_relevance([i])
                    else:
                        edge_rel[(i,j)] = self.subgraph_relevance([i,j])
                        edge_rel[(i,j)] -= self.subgraph_relevance([i]) + self.subgraph_relevance([j])

                    # Turn it into float
                    edge_rel[(i,j)] = float(edge_rel[(i,j)])
        return edge_rel

    def visit_relevance(self, nodes, inter_union='union', from_walks=False):
        assert len(nodes) == len(set(nodes)), f'Nodes are {nodes}. No node doubling please.'
        if from_walks:
            # Check whether walks has been computed
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens') # just build the tensor

            R_out = 0
            # iterate over all walk
            walks = self.walk_rels_tens.nonzero()
            # todo: itertools - select walks in the forehand
            for walk in walks:
                if inter_union=='union' and any([I in walk for I in nodes]):
                    R_out += self.walk_rels_tens[tuple(walk)]
                if inter_union=='inter' and all([I in walk for I in nodes]):
                    R_out += self.walk_rels_tens[tuple(walk)]

            return R_out
        else:
            full_r = self.subgraph_relevance(self.node_domain)
            ncomp_r = self.subgraph_relevance( list(set(self.node_domain) - set(nodes)))

            if inter_union=='union':
                return full_r - ncomp_r
            if inter_union=='inter':
                return full_r - ncomp_r - self.subgraph_relevance( nodes)

    def subgraph_relevance(self, subgraph, from_walks=False):
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens') # just build the tensor

            # Transform subgraph which is given by a set of node representation,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn: m[ft] = 1
            ms = [m]*self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0]*(self.num_layer-1-dim) + [-1]*dim:
                     ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensorproduct
            m = reduce( lambda x,y: x*y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Justs sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph*self.scal_val
        else:
            # Initialize the last relevance
            curr_subgraph_node = Node(
                            0,
                            self.lamb_per_layer[self.num_layer -1],
                            None,
                            self.R_T[0] if not self.batch_dim else self.R_T[0,0],
                            domain_restrict=None
                            )

            # self._update_tree_nodes(self.num_layer, self.R_T, 0 , None)

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes
                R = self._relprop_standard(act,
                                        layer,
                                        curr_subgraph_node.R,
                                        curr_subgraph_node.node_rep)

                # Create new subgraph nodes
                new_node = Node(subgraph,
                                self.lamb_per_layer[layer_id -1],
                                curr_subgraph_node,
                                R[subgraph] if not self.batch_dim else R[0,subgraph],
                                domain_restrict=None
                                )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum()*self.scal_val

    def walk_relevance(self, verbose=False, rel_rep='list'):
        '''
        An interface to reach for the relevance scores of all walks.
        '''
        # import pdb; pdb.set_trace()
        if not self.walk_rels_computed:
            if verbose: print('setting up walk relevances for the full graph.. this may take a wile.')
            self.setup_walk_relevance_scores()

        # Just return all walk relevances
        if rel_rep == 'tens':
            # ask for tensor representation
            if self.walk_rels_tens is None: # Not prepared yet
                self.walk_rels_tens = torch.zeros((len(self.walk_rel_domain),)*len(self.layers))
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()

                    walk_idns = tuple([ self.node2idn[idn] for idn in walk ])
                    self.walk_rels_tens[walk_idns] = rel*self.scal_val

            return self.walk_rels_tens, self.node2idn
        elif rel_rep == 'list': # ask for list representation
            if self.walk_rels_list is None: # not prepared yet
                self.walk_rels_list = []
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()
                    self.walk_rels_list.append((walk, rel * self.scal_val))

            return self.walk_rels_list


def setup_dep_graph(sample, model, target_property,cutoff=None, new_model=True):
    """ Function that sets up a dependency graph and its relevance scores for SchNet model."""
    if new_model:
        # Preprocess data sample
        _, n_atoms, _, idx_i, idx_j, x, _, f_ij, rcut_ij, node_range, lamb = get_prepro_sample(sample, model, new_model=new_model)

        ## Define layer in a list
        layers = [ (lambda h, curr_layer=inter: (h + curr_layer(h, f_ij, idx_i, idx_j, rcut_ij)))
                    for inter in model.representation.interactions ]

        def out_layer(h): sample['scalar_representation'] = h; return model.output_modules[0](sample)[target_property]
        layers += [out_layer]
    else:
        atomic_numbers, n_atoms, r_ij, neighbors, neighbor_mask, f_ij, x, node_range, lamb = get_prepro_sample(sample, model, new_model=new_model, cutoff=cutoff)
        # Define layer in a list
        layers = [ lambda h, curr_layer=interaction: h + curr_layer( h, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
                   for interaction in model.representation.interactions ]
        def out_layer(h): sample['representation'] = h; return model.output_modules[0](sample)[target_property]
        layers += [out_layer]

    # Create the dependency graph for the relevance model
    dep_graph = SymbXAI(layers,
                                x.data,
                                n_atoms,
                                lamb,
                                R_T = None,
                                batch_dim = not new_model)
    dep_graph.setup_walk_relevance_scores()

    return dep_graph

class TransformerSymbXAI(SymbXAI):
    def __init__(self):
        pass

# todo: if the schnet is trained on forces, the positions get a gradients
# if we compute the relevance scores we get an
# "RuntimeError: Trying to backward through the graph a second time"
# I set .backward(retain_graph=True) for now in _relprop_standard. But
# this shouldn't be a long time solution
class SchNetDepGraph(SymbXAI):
    def __init__(self, sample,
            model,
            target_property,
            xai_mod=True,
            gamma = 0.1,
            cutoff=None,
            new_model=True,
            comp_domain= None,
            scal_val = 1.):
        # When computing forces, the model still has the gradients.
        model.zero_grad()
        if new_model:
            # Preprocess data sample
            _, n_atoms, _, idx_i, idx_j, x, _, f_ij, rcut_ij, node_range, lamb = get_prepro_sample(sample, model, new_model=new_model)
            # make the layer explainable
            for layer in model.representation.interactions: layer._set_xai(xai_mod, gamma)
            model.output_modules[0]._set_xai(xai_mod,gamma)

            ## Define layer in a list
            layers = []
            for inter in model.representation.interactions:
                def layer(h, curr_layer=inter):
                    curr_layer.zero_grad()
                    return h  + curr_layer(h, f_ij, idx_i, idx_j, rcut_ij)
                # layer = (lambda h, curr_layer=inter: ))
                layers.append(layer)


            def out_layer(h):
                sample['scalar_representation'] = h;
                layer = model.output_modules[0]
                layer.zero_grad()
                return layer(sample)[target_property]
            layers += [out_layer]


        else:
            assert False, 'outdated'
            atomic_numbers, n_atoms, r_ij, neighbors, neighbor_mask, f_ij, x, node_range, lamb = get_prepro_sample(sample, model, new_model=new_model, cutoff=cutoff)

            # Define layer in a list
            layers = [ lambda h, curr_layer=interaction: h + curr_layer( h, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
                     for interaction in model.representation.interactions ]

            def out_layer(h): sample['representation'] = h; return model.output_modules[0](sample)[target_property]
            layers += [out_layer]

        if comp_domain is not None:
            # just select ligand and neighbors
            graph_domain = comp_domain
        else:
            # take full range
            graph_domain = node_range

        super().__init__(layers,
                         x.data,
                         n_atoms,
                         lamb,
                         R_T = None,
                         batch_dim = not new_model,
                         scal_val=scal_val)


def get_prepro_sample(sample,model,new_model=True, add_selfconn=True, cutoff=None):
    if new_model:
        if spk.properties.Rij not in sample:
            model(sample) # set up some parameter in sample
        atomic_numbers = sample[spk.properties.Z]
        r_ij = sample[spk.properties.Rij]
        idx_i = sample[spk.properties.idx_i]
        idx_j = sample[spk.properties.idx_j]
        n_atoms = sample[spk.properties.n_atoms]

        # compute atom and pair features
        x = model.representation.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1).float() ## Hack in SchNet
        f_ij = model.representation.radial_basis(d_ij)
        rcut_ij = model.representation.cutoff_fn(d_ij)

        node_range= [i for i in range(n_atoms[0])]
        lamb = torch.zeros(n_atoms[0], n_atoms[0])

        if cutoff is None:
            # just set the given connections
            lamb[idx_i, idx_j] = 1
        else:
            for i,j,d in zip(idx_i, idx_j, d_ij):
                if d<=cutoff: lamb[i,j] =1

        if add_selfconn: lamb += torch.eye(n_atoms[0])

        return atomic_numbers, n_atoms, r_ij, idx_i, idx_j, x, d_ij, f_ij, rcut_ij, node_range, lamb

    else:

        # get tensors from input dictionary
        atomic_numbers = sample[spk.Properties.Z]
        positions = sample[spk.Properties.R]
        cell = sample[spk.Properties.cell]
        cell_offset = sample[spk.Properties.cell_offset]
        neighbors = sample[spk.Properties.neighbors]
        neighbor_mask = sample[spk.Properties.neighbor_mask]
        atom_mask = sample[spk.Properties.atom_mask]
        n_atoms = torch.tensor(atomic_numbers.shape[1]).unsqueeze(0)

        x = model.representation.embedding(atomic_numbers)

        # compute interatomic distance of every atom to its neighbors
        r_ij = model.representation.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = model.representation.distance_expansion(r_ij)


        node_range= [i for i in range(n_atoms[0])]

        # compose the adjacency matrix aka lambda
        hard_cutoff_network = spk.nn.cutoff.HardCutoff(cutoff)
        lamb_raw = hard_cutoff_network(r_ij)[0]

        lamb = torch.zeros(lamb_raw.shape[0], lamb_raw.shape[1] + 1)

        for row_idx, row in enumerate(lamb_raw):
            lamb[row_idx] = torch.cat((row[:row_idx], torch.tensor([0.]), row[row_idx:]))

        if add_selfconn: lamb += torch.eye(n_atoms[0])

        return atomic_numbers, n_atoms, r_ij, neighbors, neighbor_mask, f_ij, x, node_range, lamb
