#!/bin/bash

CHECKPOINT_PATH=../../checkpoints/gpt2
MPSIZE=1
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=512

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

WORLD_SIZE=2 python3 generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --tokenizer-type RubertaBPETokenizer \
       --tokenizer-path ../../data/vocab_50000.bpe \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --sample-input-file ./input.txt \
       --num-samples 0 \
       --top_p $TOPP \
       --recompute
