from utils_new import *

def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                    HW_row_norm = False):
    '''
        FastGCN_Sampler: 
        Sample a fixed number of nodes per layer. The sampling probability (importance)
        is pre-computed based on the global degree (lap_matrix)
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    #     pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. 
        Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.         
        # adj = row_norm(U[: , after_nodes].multiply(1/p[after_nodes]))
        adj = U[: , after_nodes].multiply(1/p[after_nodes]/s_num)
        #     Turn the sampled adjacency matrix into a sparse matrix. If implemented by PyG
        #     This sparse matrix can also provide index and value.
        # adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #   Reverse the sampled probability from bottom to top. 
    #   Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, []

def fastgcn_sampler_custom(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, 
                    depth, HW_row_norm = False, flat=False, wrs=False):
    '''
        FastGCN_Sampler: 
        Sample a fixed number of nodes per layer. The sampling probability (importance)
        is pre-computed based on the global degree (lap_matrix)
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    # pre-compute the sampling probability (importance) based on 
    # the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    if flat: pi = np.sqrt(pi)
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. 
        Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])

        if wrs:
            after_nodes, weights = estWRS_weights(p, s_num)
            adj = U[: , after_nodes].multiply(weights)
        else:
            after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
            adj = U[: , after_nodes].multiply(1/p[after_nodes]/s_num)
        
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        previous_nodes = after_nodes

    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, []

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                   HW_row_norm = False):
    '''
        LADIES_Sampler: 
        Sample a fixed number of nodes per layer. The sampling probability (importance)
        is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively 
        (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]

        # pi= np.array(np.sqrt(np.sum(U.multiply(U), axis=0)))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop

        # LADIES's original codes add batch_nodes into after_nodes.
        # This means it samples more nodes than reported nodes.
        # To make comparision fair, we remove the concatenation operation
        # after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))

        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion. 
        # remove rownormalize in orginal codes. Instead, normalize U by / s_sum     
        # adj = U[: , after_nodes].multiply(1 / p[after_nodes])
        # adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        adj = U[: , after_nodes].multiply(1 / p[after_nodes] / s_num)
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #   Reverse the sampled probability from bottom to top.  
    #   Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, []

def ladies_sampler_wrs(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                   HW_row_norm = False):
    '''
        LADIES_Sampler: 
        Sample a fixed number of nodes per layer. The sampling probability (importance)
        is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively 
        (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]

        # pi= np.array(np.sqrt(np.sum(U.multiply(U), axis=0)))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes, weights = estWRS_weights(p, s_num)
        adj = U[: , after_nodes].multiply(weights)
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #   Reverse the sampled probability from bottom to top.  
    #   Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, []

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                    HW_row_norm = False):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes, []

def full_batch_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                       HW_row_norm = False):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    mx_top = sparse_mx_to_torch_sparse_tensor(lap_matrix[batch_nodes, :])
    adjs = [mx for i in range(depth -1)]
    adjs.append(mx_top)
    return adjs, np.arange(num_nodes), batch_nodes, []


def full_batch_sampler_eco(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth, HW_row_norm = False):
    # mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    adjs = []
    mx = lap_matrix.T
    # print(type(mx))
    prev_nodes = batch_nodes
    for i in range(depth):
        adj = mx[prev_nodes, :]
        next_nodes = np.unique(adj.indices)
        adjs.append(adj[:, next_nodes].tocoo())
        # print(adjs[-1].shape)
        prev_nodes = next_nodes
    
    # print(type(adjs[0]))
    adjs.reverse()
    # print(type(adjs[0]))
    # adjs = [[torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64)), 
    #     torch.from_numpy(adj.data.astype(np.float32)), 
    #     torch.Size(adj.shape)] for adj in adjs]
    # adjs = [[np.vstack((adj.row, adj.col)).astype(np.int64), 
    #     adj.data.astype(np.float32), adj.shape] for adj in adjs]
    # adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
    # print(type(adjs[0]))
    return (adjs, prev_nodes, batch_nodes, [])

def sketch_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                    HW_row_norm = False):
    '''
        Sketch Sampler:
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    after_nodes_ls = []
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        # pi = np.array(np.sum(U.multiply(U), axis=0))[0]

        #   Weights for the importance sampling:
        #       combine colNorm of U[previous_nodes, :] and historical rowNorm of HW
        pi  = np.sqrt(np.array(np.sum(U.multiply(U), axis=0))[0])

        #   if there is rowNorm infomation, then intergate it
        #   depth - 1 - d's layer because we do reverse sampling
        if not (HW_row_norm is False):
            pi2 = HW_row_norm[depth-1-d, : ]
            pi  = pi * pi2
        
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop

        # LADIES's original codes add batch_nodes into after_nodes.
        # This means it samples more nodes than reported nodes.
        # To make comparision fair, we remove the concatenation operation
        # after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))

        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion. 
        # remove rownormalize in orginal codes. Instead, normalize U by / s_sum     
        adj = U[: , after_nodes].multiply(1/p[after_nodes] / s_num)
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        
        previous_nodes = after_nodes
        after_nodes_ls.append(after_nodes.copy())
        
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    after_nodes_ls.reverse()
    #   arg: after_nodes is the after_nodes for most inside layer
    #       i.e. used for  H1[after_nodes, :] @ W1 
    #       i.e. the input nodes for most inside layer
    #   arg: batch_nodes: the output nodes (for most outside layer)
    return adjs, previous_nodes, batch_nodes, after_nodes_ls
    
def sketch_sampler_wrs(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth,
                    HW_row_norm = False):
    '''
        Sketch Sampler:
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    after_nodes_ls = []
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        # pi = np.array(np.sum(U.multiply(U), axis=0))[0]

        #   Weights for the importance sampling:
        #       combine colNorm of U[previous_nodes, :] and historical rowNorm of HW
        pi  = np.sqrt(np.array(np.sum(U.multiply(U), axis=0))[0])

        #   if there is rowNorm infomation, then intergate it
        #   depth - 1 - d's layer because we do reverse sampling
        if not (HW_row_norm is False):
            pi2 = HW_row_norm[depth-1-d, : ]
            pi  = pi * pi2
        
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes, weights = estWRS_weights(p, s_num)
        adj = U[: , after_nodes].multiply(weights)
        
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        
        previous_nodes = after_nodes
        after_nodes_ls.append(after_nodes.copy())
        
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    after_nodes_ls.reverse()
    #   arg: after_nodes is the after_nodes for most inside layer
    #       i.e. used for  H1[after_nodes, :] @ W1 
    #       i.e. the input nodes for most inside layer
    #   arg: batch_nodes: the output nodes (for most outside layer)
    return adjs, previous_nodes, batch_nodes, after_nodes_ls
    

def prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes,
                 lap_matrix, depth, batch_size, HW_row_norm = False, full_nbar = True):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                             samp_num_list, num_nodes, lap_matrix, depth, HW_row_norm))
        jobs.append(p)
    # idx = torch.randperm(len(valid_nodes))[:batch_size]
    # use all validation nodes with full batch-inference
    batch_nodes = valid_nodes

    if full_nbar:
        # valid_sampler = full_batch_sampler
        valid_sampler = full_batch_sampler_eco
    else:
        valid_sampler = sampler

    p = pool.apply_async(valid_sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                         samp_num_list * 20, num_nodes, lap_matrix, depth, HW_row_norm))
    jobs.append(p)
    return jobs

def prepare_data_ose(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes,
                     lap_matrix, depth, batch_size, s, m, HW_row_norm = False, full_nbar = True):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                             lap_matrix, s, m, depth))
        jobs.append(p)
    # idx = torch.randperm(len(valid_nodes))[:batch_size]
    # use all validation nodes with full batch-inference
    batch_nodes = valid_nodes

    if full_nbar:
        p = pool.apply_async(full_batch_sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                          samp_num_list * 20, num_nodes, lap_matrix, depth, HW_row_norm))
    else:
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                             lap_matrix, s, m, depth))

    
    jobs.append(p)
    return jobs
