import torch
from functools import reduce
import numpy as np
from ..model.transformer import ModifiedTinyTransformerForSequenceClassification,  ModifiedBertForSequenceClassification
import schnetpack as spk


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
        if isinstance(lamb, list):
            self.lamb_per_layer = lamb
        else:
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

    def _setup_walk_relevance_scores(
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
            self
    ):

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

        return node_rel

    def symb_or(self,
        featset,
        context = None
    ):
        if context is None:
            context = self.node_domain

        return self.subgraph_relevance( context ) - \
                self.subgraph_relevance(
                        list(set(context) - set(featset)))


    def symb_not(self,
        featset,
        context=None):

        if context is None:
            context = self.node_domain

        return self.subgraph_relevance( context ) - \
                self.symb_or(featset, context=context)

    def symb_and(self,
        featset,
        context=None
    ):
        assert len(featset) <=2, 'Sorry, the "and" operator for more than 2 ' \
                            +'elements is not implemented yet!'

        if len(featset) <= 1:
            return self.symb_or(featset,context=context)

        elif  len(featset) == 2:
            s1, s2 = [featset[0]], [featset[1]]

            return self.symb_or(s1,context=context) \
                    + self.symb_or(s2,context=context)  \
                    - self.symb_or(featset,context=context)


    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        if type(subgraph) != list:
            subgraph = list(subgraph)

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
            if verbose:
                print('setting up walk relevances for the full graph.. this may take a wile.')
            self._setup_walk_relevance_scores()

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
        lamb_last_layer = torch.zeros((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.bert.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            pooled_data = modified_model.bert.pooler(hidden_states)

            output = (modified_model.classifier(pooled_data) * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)

        lamb_last_layer[0, :] = torch.ones(num_tokens)
        lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]


        super().__init__(
            layers,
            x.data,
            num_tokens,
            lambs,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        if type(subgraph) != list:
            subgraph = list(subgraph)
        # TODO: Change the code for from_walks=True
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
                    new_node = Node(0,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0],
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

    def get_local_best_subgraph(
            self,
            alpha: float = 0.0
    ):
        subgraph = []
        all_features = np.arange(self.num_nodes)

        while len(subgraph) < self.num_nodes:
            feature_list = list(frozenset(all_features).difference(frozenset(subgraph)))
            max_score = -float("inf")
            max_feature = None

            graph_score = self.subgraph_relevance(subgraph=all_features, from_walks=False)

            for feature in feature_list:
                # s = subgraph + [feature]
                s = list(frozenset(all_features).difference(frozenset(subgraph + [feature])))

                # mask = torch.full([self.num_nodes], alpha)
                # mask[list(s)] = 1.0
                # mask = torch.diag(mask)

                temp_score = -np.abs(self.subgraph_relevance(subgraph=list(s), from_walks=False) - graph_score)

                if temp_score > max_score:
                    max_score = temp_score
                    max_feature = feature

            subgraph += [max_feature]

        best_subgraph = torch.full((self.num_nodes, ), -1)
        for i, f in enumerate(subgraph):
            best_subgraph[f] = self.num_nodes - i

        return best_subgraph


class BERTSymbXAI(SymbXAI):
    def __init__(
            self,
            sample,
            target,
            model,
            embeddings,
            scal_val=1.,
            use_lrp_layers=True
    ):
        model.zero_grad()

        # Prepare the input embeddings.
        x = embeddings(
            input_ids=sample['input_ids'],
            token_type_ids=sample['token_type_ids']
        )

        # Make the model explainable.

        if use_lrp_layers:
            modified_model = ModifiedBertForSequenceClassification(
                model
            )
        else:
            modified_model = model

        if len(x.shape) >= 3:
            batch_dim = True
            num_tokens = x.shape[1]
        else:
            batch_dim = False
            num_tokens = x.shape[0]

        lamb = torch.ones((num_tokens, num_tokens))
        lamb_last_layer = torch.zeros((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.bert.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            pooled_data = modified_model.bert.pooler(hidden_states)
            logits = modified_model.classifier(pooled_data)
            output = (logits * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)

        lamb_last_layer[0, :] = torch.ones(num_tokens)
        lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]

        super().__init__(
            layers,
            x.data,
            num_tokens,
            lambs,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):

        # TODO: Change the code for from_walks=True
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

                if layer_id == 12:
                    # Create new subgraph nodes.
                    new_node = Node(0,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0],
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

    def subgraph_shap(
            self,
            subgraph
    ):
        out = self.xs[0][subgraph].unsqueeze(0) if not self.batch_dim else self.xs[0][0, subgraph].unsqueeze(0)
        for layer in self.layers:
            out = layer(out)[0]

        return out * self.scal_val

    def symb_or_shap(
            self,
            featset,
            context=None
            ):
        if context is None:
            context = self.node_domain

        return self.subgraph_shap(context) - \
               self.subgraph_shap(
                   list(set(context) - set(featset)))


######################
# Quantum Chemistry #
#####################
class SchNetSymbXAI(SymbXAI):
    def __init__(
        self,
        sample,
        model,
        target_property,
        xai_mod=True,
        gamma=0.1,
        cutoff=None,
        new_model=True,
        comp_domain=None,
        scal_val=1.
    ):
        model.zero_grad()  # When computing forces, the model still has the gradients.
        _, n_atoms, _, idx_i, idx_j, x, _, f_ij, rcut_ij, node_range, lamb = get_prepro_sample_qc(
            sample, model, new_model=new_model
        )

        for layer in model.representation.interactions:
            layer._set_xai(xai_mod, gamma)
        model.output_modules[0]._set_xai(xai_mod, gamma)

        layers = []
        for inter in model.representation.interactions:
            def layer(h, curr_layer=inter):
                curr_layer.zero_grad()
                return h + curr_layer(h, f_ij, idx_i, idx_j, rcut_ij)
            layers.append(layer)

        def out_layer(h):
            sample['scalar_representation'] = h
            layer = model.output_modules[0]
            layer.zero_grad()
            return layer(sample)[target_property]
        layers += [out_layer]

        super().__init__(
            layers,
            x.data,
            n_atoms,
            lamb,
            R_T=None,
            batch_dim=not new_model,
            scal_val=scal_val
        )


def get_prepro_sample_qc(
    sample,
    model,
    new_model=True,
    add_selfconn=True,
    cutoff=None
):
    if new_model:
        if spk.properties.Rij not in sample:
            model(sample)

        atomic_numbers = sample[spk.properties.Z]
        r_ij = sample[spk.properties.Rij]
        idx_i = sample[spk.properties.idx_i]
        idx_j = sample[spk.properties.idx_j]
        n_atoms = sample[spk.properties.n_atoms]

        x = model.representation.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1).float()
        f_ij = model.representation.radial_basis(d_ij)
        rcut_ij = model.representation.cutoff_fn(d_ij)

        node_range = [i for i in range(n_atoms[0])]
        lamb = torch.zeros(n_atoms[0], n_atoms[0])

        if cutoff is None:
            lamb[idx_i, idx_j] = 1
        else:
            for i, j, d in zip(idx_i, idx_j, d_ij):
                if d <= cutoff:
                    lamb[i, j] = 1

        if add_selfconn:
            lamb += torch.eye(n_atoms[0])

        return (
            atomic_numbers,
            n_atoms,
            r_ij,
            idx_i,
            idx_j,
            x,
            d_ij,
            f_ij,
            rcut_ij,
            node_range,
            lamb
        )
    else:
        atomic_numbers = sample[spk.Properties.Z]
        positions = sample[spk.Properties.R]
        cell = sample[spk.Properties.cell]
        cell_offset = sample[spk.Properties.cell_offset]
        neighbors = sample[spk.Properties.neighbors]
        neighbor_mask = sample[spk.Properties.neighbor_mask]
        atom_mask = sample[spk.Properties.atom_mask]
        n_atoms = torch.tensor(atomic_numbers.shape[1]).unsqueeze(0)

        x = model.representation.embedding(atomic_numbers)
        r_ij = model.representation.distances(
            positions,
            neighbors,
            cell,
            cell_offset,
            neighbor_mask=neighbor_mask
        )
        f_ij = model.representation.distance_expansion(r_ij)
        node_range = [i for i in range(n_atoms[0])]

        hard_cutoff_network = spk.nn.cutoff.HardCutoff(cutoff)
        lamb_raw = hard_cutoff_network(r_ij)[0]

        lamb = torch.zeros(lamb_raw.shape[0], lamb_raw.shape[1] + 1)

        for row_idx, row in enumerate(lamb_raw):
            lamb[row_idx] = torch.cat((row[:row_idx], torch.tensor([0.]), row[row_idx:]))

        if add_selfconn:
            lamb += torch.eye(n_atoms[0])

        return (
            atomic_numbers,
            n_atoms,
            r_ij,
            neighbors,
            neighbor_mask,
            f_ij,
            x,
            node_range,
            lamb
        )
