import torch
from functools import reduce
import numpy as np
from model.transformer import ModifiedTinyTransformerForSequenceClassification


class Node:
    """
    Helping class for the explainer functions.
    """

    def __init__(
            self,
            node_rep,
            lamb,
            parent,
            R,
            domain_restrict=None
    ):
        self.node_rep = node_rep
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
    def __init__(
            self,
            layers,
            x,
            num_nodes,
            lamb,
            R_T=None,
            batch_dim=False,
            scal_val=1.
    ):
        """
        Init function. It basically sets some hyperparameters and saves the activations.
        """
        # Save some parameter.
        self.layers = layers
        self.num_layer = len(layers)
        self.node_domain = list(range(num_nodes))
        self.num_nodes = num_nodes
        self.batch_dim = batch_dim
        self.scal_val = scal_val

        # Some placeholder parameter for later.
        self.tree_nodes_per_act = None
        self.walk_rels_tens = None
        self.walk_rels_list = None
        self.node2idn = None
        self.walk_rel_domain = None
        self.walk_rels_computed = False

        # Set up activations.
        self.xs = [x.data]
        for layer in layers:
            x = layer(x)
            self.xs.append(x.data)

        # Set up for each layer the dependencies.
        self.lamb_per_layer = [lamb for _ in range(self.num_layer - 1)] + [torch.ones(self.num_nodes).unsqueeze(0)]

        # Initialize the relevance.
        if R_T is None:
            self.R_T = self.xs[-1].data.detach()
        else:
            self.R_T = R_T

    def _relprop_standard(
            self,
            act,
            layer,
            R,
            node_rep
    ):
        """
        Just a vanilla relevance propagation strategy per layer guided along the specified nodes.
        """
        act = act.data.requires_grad_(True)

        # Forward layer guided at the node representation.
        z = layer(act)[node_rep] if not self.batch_dim else layer(act)[0, node_rep]

        assert z.shape == R.shape, f'z.shape {z.shape}, R.shape {R.shape}'

        s = R / z
        (z * s.data).sum().backward(retain_graph=True)
        c = torch.nan_to_num(act.grad)
        R = act * c

        return R.data

    def _update_tree_nodes(
            self,
            act_id,
            R,
            node_rep,
            parent,
            domain_restrict=None
    ):
        """
        Update the nodes in the internal dependency tree, by the given hyperparameter.
        """
        lamb = self.lamb_per_layer[act_id - 1]
        self.tree_nodes_per_act[act_id] += [Node(node_rep,
                                                 lamb,
                                                 parent,
                                                 R[node_rep] if not self.batch_dim else R[0, node_rep],
                                                 domain_restrict=domain_restrict)]

    def setup_walk_relevance_scores(
            self,
            domain_restrict=None,
            verbose=False
    ):
        """
        To set up the relevance scores in the quality of walks. All scores will be saved at self.walk_rels_tens
        and self.walk_rels_list.
        """

        # Create data structure.
        self.tree_nodes_per_act = [[] for _ in range((self.num_layer + 1))]

        # Initialize the last relevance activation.
        self._update_tree_nodes(
            self.num_layer,
            self.R_T,
            0,
            None,
            domain_restrict=domain_restrict
        )

        for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
            # Iterate over the nodes.
            for node in self.tree_nodes_per_act[layer_id + 1]:
                # Compute the relevance.
                R = self._relprop_standard(
                    act,
                    layer,
                    node.R,
                    node.node_rep
                )


                # Distribute the relevance to the neighbors.
                for neigh_rep in node.neighbors():
                    self._update_tree_nodes(
                        layer_id,
                        R,
                        neigh_rep,
                        node,
                        domain_restrict=domain_restrict
                    )

        # save a few parameters.
        if domain_restrict is None:
            self.walk_rel_domain = self.node_domain
        else:
            self.walk_rel_domain = domain_restrict
        self.node2idn = {node: i for i, node in enumerate(self.walk_rel_domain)}
        self.walk_rels_computed = True


    def node_relevance(
            self,
            mode='DeltaR'
    ):

        if mode == 'node@input':
            # Initialize the last relevance.
            curr_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(
                    act,
                    layer,
                    curr_node.R,
                    curr_node.node_rep
                )

                # Create new nodes
                new_node = Node(
                    self.node_domain,
                    self.lamb_per_layer[layer_id - 1],
                    curr_node,
                    R[self.node_domain] if not self.batch_dim else R[0, self.node_domain],
                    domain_restrict=None
                )

                curr_node = new_node
            node_rel = curr_node.R.sum(-1) * self.scal_val
        elif mode == 'DeltaR':
            node_rel = np.array([self.subgraph_relevance([node]) for node in self.node_domain])

        return node_rel

    def edge_relevance(
            self,
            mode='DeltaR',
            cust_edges=None,
            with_selfloop=True,
            from_walks=False
    ):
        assert mode in ['edge_in_walk', 'walk_in_edge', 'edge@input', 'DeltaR'], f'mode "{mode}" is not defined'

        if mode == 'edge@input':
            # Initialize the last relevance
            curr_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                if layer_id > 0:
                    # Simple first order backpropagation.
                    R = self._relprop_standard(act,
                                               layer,
                                               curr_node.R,
                                               curr_node.node_rep)

                    # Create new nodes
                    new_node = Node(
                        self.node_domain,
                        self.lamb_per_layer[layer_id - 1],
                        curr_node,
                        R[self.node_domain] if not self.batch_dim else R[0, self.node_domain],
                        domain_restrict=None
                    )

                    curr_node = new_node
                else:
                    if cust_edges is None:
                        lamb = self.lamb_per_layer[layer_id]
                    else:
                        lamb = torch.zeros(self.lamb_per_layer[layer_id].shape)
                        for i, j in cust_edges: lamb[i, j] = 1

                    temp_node_domain = set(lamb.nonzero().T.numpy()[0])
                    init_nodes = [Node(node_rep,
                                       lamb,
                                       None,
                                       curr_node.R[node_rep] if not self.batch_dim else R[0, node_rep])
                                  for node_rep in temp_node_domain]
                    out_nodes = []
                    # Iterate over the nodes
                    for node in init_nodes:
                        # Compute the relevance
                        R = self._relprop_standard(
                            act,
                            layer,
                            node.R,
                            node.node_rep
                        )

                        # Distribute the relevance to the neighbors
                        for neigh_rep in node.neighbors():
                            out_nodes += [Node(neigh_rep,
                                               None,
                                               node,
                                               R[neigh_rep] if not self.batch_dim else R[0, neigh_rep])]

            # Extract the edge relevance.
            edge_rel = {}

            for node in out_nodes:
                walk, rel = node.get_walk(), node.R.data.sum().item()
                edge_rel[walk] = rel * self.scal_val

        elif mode == 'edge_in_walk':
            # Get edges.
            edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]

            wrel = self.walk_relevance(verbose=False)
            edge_rel = {}
            for edge in edges:
                erel = 0.
                for w, rel in wrel:
                    for t in range(len(w) - 1):
                        if w[t:t + 2] == edge:
                            erel += rel
                            break
                edge_rel[edge] = erel.item() / (self.num_layer - 1)

        elif mode == 'walk_in_edge':
            # Get edges.
            edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]

            edge_rel = {edge: self.subgraph_relevance(edge).item() for edge in edges}

        elif mode == 'DeltaR':
            if cust_edges is None:
                # Get edges.
                edges = [tuple(edge) for edge in self.lamb_per_layer[0].nonzero().numpy()]
                if not with_selfloop:
                    edges = [(i, j) for (i, j) in edges if i != j]
            else:
                edges = cust_edges
            edge_rel = {}

            if from_walks:
                wrel = self.walk_relevance(verbose=False)
                for (i, j) in edges:
                    edge_rel[(i, j)] = 0.
                    for walk, rel in wrel:
                        if i != j and len(set(walk)) == 2 and all([w in [i, j] for w in walk]):
                            edge_rel[(i, j)] += float(rel)
                        elif i == j and len(set(walk)) == 1 and all([w in [i, j] for w in walk]):
                            edge_rel[(i, j)] += float(rel)
                        else:
                            continue
            else:
                for (i, j) in edges:
                    if i == j:
                        edge_rel[(i, j)] = self.subgraph_relevance([i])
                    else:
                        edge_rel[(i, j)] = self.subgraph_relevance([i, j])
                        edge_rel[(i, j)] -= self.subgraph_relevance([i]) + self.subgraph_relevance([j])

                    # Turn it into float.
                    edge_rel[(i, j)] = float(edge_rel[(i, j)])
        return edge_rel

    def visit_relevance(
            self,
            nodes,
            inter_union='union',
            from_walks=False
    ):
        assert len(nodes) == len(set(nodes)), f'Nodes are {nodes}. No node doubling please.'
        if from_walks:

            # Check whether walks have been computed.
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            R_out = 0
            # Iterate over all walks.
            walks = self.walk_rels_tens.nonzero()
            # TODO: itertools - select walks in the forehand.
            for walk in walks:
                if inter_union == 'union' and any([I in walk for I in nodes]):
                    R_out += self.walk_rels_tens[tuple(walk)]
                if inter_union == 'inter' and all([I in walk for I in nodes]):
                    R_out += self.walk_rels_tens[tuple(walk)]

            return R_out
        else:
            full_r = self.subgraph_relevance(self.node_domain)
            ncomp_r = self.subgraph_relevance(list(set(self.node_domain) - set(nodes)))

            if inter_union == 'union':
                return full_r - ncomp_r
            if inter_union == 'inter':
                return full_r - ncomp_r - self.subgraph_relevance(nodes)

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:
            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                # Create new subgraph nodes.
                new_node = Node(subgraph,
                                self.lamb_per_layer[layer_id - 1],
                                curr_subgraph_node,
                                R[subgraph] if not self.batch_dim else R[0, subgraph],
                                domain_restrict=None
                                )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val

    def walk_relevance(self, verbose=False, rel_rep='list'):
        """
        An interface to reach for the relevance scores of all walks.
        """

        if not self.walk_rels_computed:
            if verbose: print('setting up walk relevances for the full graph.. this may take a wile.')
            self.setup_walk_relevance_scores()

        # Just return all walk relevances.
        if rel_rep == 'tens':
            # Ask for tensor representation.
            if self.walk_rels_tens is None:  # Not prepared yet.
                self.walk_rels_tens = torch.zeros((len(self.walk_rel_domain),) * len(self.layers))
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()

                    walk_idns = tuple([self.node2idn[idn] for idn in walk])
                    self.walk_rels_tens[walk_idns] = rel * self.scal_val

            return self.walk_rels_tens, self.node2idn
        elif rel_rep == 'list':  # Ask for list representation.
            if self.walk_rels_list is None:  # Not prepared yet.
                self.walk_rels_list = []
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()
                    self.walk_rels_list.append((walk, rel * self.scal_val))

            return self.walk_rels_list


class TransformerSymbXAI(SymbXAI):
    def __init__(
            self,
            sample,
            target,
            model,
            embeddings,
            scal_val=1.
    ):
        model.zero_grad()

        # Prepare the input embeddings.
        x = embeddings(
            input_ids=sample['input_ids'],
            token_type_ids=sample['token_type_ids']
        )

        # Make the model explainable.
        modified_model = ModifiedTinyTransformerForSequenceClassification(
            model,
            order='first'
        )

        if len(x.shape) >= 3:
            batch_dim = True
            num_tokens = x.shape[1]
        else:
            batch_dim = False
            num_tokens = x.shape[0]

        lamb = torch.ones((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.bert.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            pooled_data = modified_model.bert.pooler(hidden_states)
            output = (modified_model.classifier(pooled_data) * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)


        super().__init__(
            layers,
            x.data,
            num_tokens,
            lamb,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:
            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                if layer_id == 3:
                    # Create new subgraph nodes.
                    new_node = Node(subgraph,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0].repeat(len(subgraph), 1),
                                    domain_restrict=None
                                    )
                else:
                    # Create new subgraph nodes.
                    new_node = Node(subgraph,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[subgraph] if not self.batch_dim else R[0, subgraph],
                                    domain_restrict=None
                                    )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val


