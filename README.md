# RuGPT2
Russian GPT-2 model

## Pretrain GPT-2 model

1. Cleanup data and preprocess json:
```
python3 scripts/cleanup_dataset.py <input_file.txt> <output_file.json>
```
2. Preprocess data for training:
```
python3 scripts/split_gpt2_json.py \
    --input_files <input_file.json> \
    --output_dir <output_dir> \
    --test_percent 0.001 \
```
3. Run training:
```
./scripts/pretrain_gpt2.sh
```

## Generate text samples
```
./scripts/generate_text_yttm_file.sh
```
