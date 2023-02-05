CUDA_VISIBLE_DEVICES=, python3 sketch/main.py \
    --cuda -1 --n_trial 1 --epoch_num 2 --record_f1 --pool_num 2 \
    --dataset ogbn-arxiv --samp_growth_rate 1 --samp_num 64
