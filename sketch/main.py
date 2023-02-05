
#!/usr/bin/env python
# coding: utf-8
# test sync in gd

from ogb.nodeproppred import Evaluator

import datetime, argparse, sys
from functools import partial
import scipy
import multiprocessing as mp

from utils_new import *
from models import GCN, SuGCN
from samplers import (fastgcn_sampler, fastgcn_sampler_custom, 
    ladies_sampler, ladies_sampler_wrs, sketch_sampler, sketch_sampler_wrs,
    prepare_data, full_batch_sampler, full_batch_sampler_eco)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='ogbn-proteins',
                    help='Dataset name: cora/citeseer/pubmed/reddit/ppi/ppi-large')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='sketch',
                    help='Sampled Algorithms: ladies/fastgcn/full/sketch')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID: -1 for cpu, 0 for gpu')
parser.add_argument('--n_trial', type=int, default=1,
                    help='Number of times to repeat experiments')
parser.add_argument('--record_f1', action='store_true',
                    help='whether record the f1 score')
parser.add_argument('--full_valid', type=int, default=1,
                    help='Use all neighbors for validation')
parser.add_argument('--samp_growth_rate', type = float, default = 2,
                    help='Growth rate for layer-wise sampling')
parser.add_argument('--full_batch', type = int, default = 0,
                    help='1: use full-batch training')
args = parser.parse_args()

args = parser.parse_args()



if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
# Load data 
#  It's time consuming to load oringinal data. we laod from pkl for convenience
print(args.dataset)
# 3) replace
if args.dataset in ["reddit", 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag']:
    with open(r'./data/{}.pkl'.format(args.dataset), "rb") as input_file:
        adj_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = pkl.load(input_file)

    print("load from pickle")
else:
    adj_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = preprocess_data(args.dataset)


# sys.exit()
n_nodes = feat_data.shape[0]

print("n train, val, test")
print(len(train_nodes), len(valid_nodes), len(test_nodes))
print("batch_size: ", args.batch_size, ", sample_size: ", args.samp_num)

# Get Laplacian (np.array): D^{-1/2} A_tilde D^{-1/2}
lap_matrix = normalize_lap(adj_matrix + sp.eye(adj_matrix.shape[0]))

if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)

#   Loss function:
#       BCEwithlogit for multi-label dataset
#       CrossEntropy for 1-label dataset
multi_label = True if args.dataset in ['ogbn-proteins'] else False
ogb_data = True if args.dataset in ['ogbn-arxiv', 'ogbn-products', 
                                        'ogbn-proteins', 'ogbn-mag'] else False

if multi_label:
    loss_func = nn.BCEWithLogitsLoss()
    labels = torch.FloatTensor(labels).to(device)
    num_classes = labels.shape[1]
else:
    loss_func = nn.CrossEntropyLoss()
    # loss_func = F.cross_entropy
    # for OGB data, we need to squeeze labels to 1D tensor. i.e. len * 1 -> len
    labels    = torch.LongTensor(labels).squeeze(1).to(device) if ogb_data else torch.LongTensor(labels).to(device)
    num_classes = labels.max().item()+1

if ogb_data:
    evaluator = Evaluator(name = args.dataset)
    print("Using OGB's accuracy criterion.")


def eval_f1(output, labels, output_nodes, num_classes, multi_label = False):
    print(args)
    if multi_label and not ogb_data:
        output[output > 0] = 1
        output[output <= 0] = 0
        lab_mat = labels[output_nodes].cpu()
        out_mat = output.cpu().detach().numpy()
        f1_per_label = np.zeros(num_classes)
        for i in range(num_classes):
            f1_per_label[i] = f1_score(lab_mat[:,i], out_mat[:,i], average='micro')

        f1_value = np.mean(f1_per_label)
    
    elif not ogb_data: # reddit
        # F1 score for 1 label (with multiple classes)
        f1_value = f1_score(labels[output_nodes].cpu(), output.argmax(dim=1).cpu(), average='micro')

    # overwrite f1 score for ogb_data. Use default evaluators
    else: 
        idx = output_nodes
        if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag']:
            y_pred = output.argmax(dim = -1)
            f1_value = evaluator.eval({'y_true': labels[idx].reshape([-1, 1]),
                                       'y_pred': y_pred.reshape([-1, 1])})['acc']
        else: # proteins
            f1_value = evaluator.eval({'y_true': labels[idx], 'y_pred': output})['rocauc'] 
    return f1_value

# sample_method_ls = ["ladies", "sketch", "ladies_wrs", "sketch_wrs"]
# sample_method_ls = ["fastgcn_flat", "fastgcn_wrs", "fastgcn_flat_wrs"]
sample_method_ls = ["sketch_wrs"]

if args.full_batch:
    sample_method_ls.append("full")
write_file = "w"
original_stdout = sys.stdout
result_pkl = dict()
result_pkl["args"] = args
best_model_idx = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')
if args.full_batch:
    filename = "main_full_{}_lay_{}_{}".format(args.dataset, args.n_layers, 
                                 best_model_idx)
else: 
    filename = "main_{}_lay_{}_{}".format(args.dataset, args.n_layers, 
                                 best_model_idx)
                                 
pool = mp.Pool(args.pool_num + 3)
for sample_method in sample_method_ls:

    if sample_method == 'ladies':
        sampler = ladies_sampler
    elif sample_method == 'fastgcn':
        sampler = fastgcn_sampler
    elif sample_method == 'fastgcn_flat':
        sampler = partial(fastgcn_sampler_custom, flat=True)
    elif sample_method == 'fastgcn_wrs':
        sampler = partial(fastgcn_sampler_custom, wrs=True)
    elif sample_method == 'fastgcn_flat_wrs':
        sampler = partial(fastgcn_sampler_custom, flat=True, wrs=True)
    elif sample_method == 'full':
        sampler = full_batch_sampler
    elif sample_method == 'sketch':
        sampler = sketch_sampler    
    elif sample_method == "ladies_wrs":
        sampler = ladies_sampler_wrs
    elif sample_method == "sketch_wrs":
        sampler = sketch_sampler_wrs

    print("Sampler: ", sample_method)
    

    # process_ids = np.arange(args.batch_num)
    process_ids = np.arange(args.pool_num)
    n_iters = args.batch_num // args.pool_num
    samp_num_list = np.array(args.samp_num * args.samp_growth_rate ** np.arange(args.n_layers), dtype = int)
    
    print("Sampler: ", sample_method, "batch_size: ", args.batch_size, "batch_num: ",
          args.batch_num, "sample_num: ", samp_num_list)

    jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, 
                        len(feat_data), lap_matrix, args.n_layers, args.batch_size)

    # record information
    all_res = []
    total_time_all = []
    test_f1_all   = []
    epoch_time_all = []
    epoch_num = []
    valid_f1_all = []
    valid_loss_all = []
    
    # gpu_warmup(device)

    for oiter in range(args.n_trial):

        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, 
                      layers=args.n_layers, dropout = 0.2).to(device)
        susage = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.2,
                          inp = feat_data.shape[1])
        susage.to(device)

        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()))
        best_val, best_tst = -1, -1
        cnt = 0
        times = []
        res   = []
        epoch_time = []
        valid_f1_single_iter = []
        valid_loss_single_iter = []
        print('-' * 10)
        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []
            '''
                Use CPU-GPU cooperation to reduce the overhead for sampling. 
                (conduct sampling while training)
            '''
            # train for one epoch
            for _iter in range(n_iters):
                print("Epoch", epoch, "before iter", len(jobs))
                train_data = [job.get() for job in jobs[:-1]]

                # print("Epoch", epoch, "after jobs.get()")

                valid_data = jobs[-1].get()

                print("Epoch", epoch, "before jobs")

                jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes,
                    samp_num_list, len(feat_data), lap_matrix, args.n_layers,
                    args.batch_size)

                print("Epoch", epoch, "after jobs")
                
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                for adjs, input_nodes, output_nodes, after_nodes_ls in train_data:
                    if args.cuda != -1: torch.cuda.synchronize()
                    t1 = time.time()

                    # print("Epoch", epoch, t1)
                    
                    adjs = package_mxl(adjs, device)
                    optimizer.zero_grad()
                    
                    susage.train()
                    output = susage.forward(feat_data[input_nodes], adjs)
                    
                    # use different losses with differnt tasks
                    # loss_train = F.cross_entropy(output, labels[output_nodes])
                    loss_train = loss_func(output, labels[output_nodes])
                    loss_train.backward()
                    # torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
                    optimizer.step()
                    
                    if args.cuda != -1: torch.cuda.synchronize()
                    t2 = time.time()
                    times += [t2 - t1]
                    train_losses += [loss_train.detach().tolist()]
                    del loss_train
                # end.record()
                # torch.cuda.synchronize()
                # print(start.elapsed_time(end))
                # print(np.sum(times))

            # perform validation at the end of each epoch
            epoch_time += [np.sum(times)]
            susage.eval()
            adjs, input_nodes, output_nodes, after_nodes_ls = valid_data
            adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
            adjs = package_mxl(adjs, device)

            # For validation, sketch method does not update HW_row_norm
            output = susage.forward(feat_data[input_nodes], adjs)

            # loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
            loss_valid = loss_func(output, labels[output_nodes]).detach().tolist()
     

            #   calculate F1 score for multi_(0, 1)_label (classes are 0 or 1)
            valid_f1   = eval_f1(output, labels, output_nodes, num_classes, multi_label)

            print(("Epoch: %d (%.4fs) Train Loss: %.2f    Valid Loss: %.2f  Valid F1: %.3f") % \
                              (epoch, sum(times[-args.batch_num:]), np.average(train_losses), loss_valid, valid_f1))
            valid_f1_single_iter.append(valid_f1)
            valid_loss_single_iter.append(loss_valid)

            if valid_f1 > best_val:
                best_val = valid_f1
                torch.save(susage, './save/best_model_{}.pt'.format(best_model_idx))
                cnt = 0
            else:
                cnt += 1
            if cnt == args.n_stops // args.batch_num:
                break

            print("Epoch", epoch, "after iter")

        del encoder, susage, optimizer, output
        torch.cuda.empty_cache()
        if args.dataset == "ogbn-products":
            device_eval = torch.device("cpu")
        else:
            device_eval = device
        best_model = torch.load('./save/best_model_{}.pt'.format(best_model_idx),
                        map_location=device_eval)
        best_model.to(device_eval)
        feat_data_eval = feat_data.to(device = device_eval)
        # print(device_eval, feat_data)

        best_model.eval()
        test_f1s = []

        '''
        If using full-batch inference for testing data:
        '''
        batch_nodes = test_nodes
        adjs, input_nodes, output_nodes, _ = full_batch_sampler_eco(
            np.random.randint(2**32 - 1), batch_nodes, samp_num_list, 
            len(feat_data_eval), lap_matrix, args.n_layers)
        adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]
        adjs = package_mxl(adjs, device_eval)
        
        # For saving the GPU memory
        output = best_model.forward(feat_data_eval[input_nodes], adjs)

        # back to the original device
        output = output.to(device)
                
        test_f1 = eval_f1(output, labels, output_nodes, num_classes, multi_label)
        test_f1s = [test_f1]
        
        print('Iteration: %d, Test F1: %.3f' % (oiter, np.average(test_f1s)))

        total_time_all += [np.sum(times)]
        test_f1_all  += [test_f1]
        epoch_num += [epoch]
        epoch_time_all += [epoch_time]
        valid_f1_all += [valid_f1_single_iter]
        valid_loss_all += [valid_loss_single_iter]

    # record F1 score and training time
    if args.record_f1:
        txt_name = filename + '.txt'
        result_pkl[sample_method] = record_result_new(args, txt_name, total_time_all, samp_num_list,
            valid_f1_all, valid_loss_all, test_f1_all, epoch_num, epoch_time_all, write_file,
            sample_method, original_stdout)
        print(sample_method, "\'s information recorded")

pool.close()
pool.join()

# record .pkl
if args.record_f1:
    with open('./result/{}/{}.pkl'.format(args.dataset, filename),'wb') as f:
        pkl.dump(result_pkl, f)
    print("All information is recorded")
