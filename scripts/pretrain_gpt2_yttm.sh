#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save ../../checkpoints/gpt2_345m \
       --load ../../checkpoints/gpt2_345m \
       --tensorboard-dir ../../logs/gpt2_345m \
       --log-interval 100 \
       --resume-dataloader \
       --train-data ../../data/train.json \
       --text-key text \
       --loose-json \
       --lazy-loader \
       --tokenizer-type RubertaBPETokenizer \
       --tokenizer-path ../../data/vocab_50000.bpe \
       --save-interval 2000 \
       --eval-interval 2000 \
       --vocab-size 50000 \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16


set +x
