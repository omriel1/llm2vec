# Connect
1. Copy the ssh command inorder to connect to the machine. Should look like
```
ssh -p 14709 root@ssh5.vast.ai -L 8080:localhost:8080
```

## First time setup

1. Create cache folders (**Just on first time**)
```
mkdir cache
mkdir llm2vec
mkdir cache/hf_cache
mkdir cache/transformers_cache
mkdir cache/hf_dataset_cache
mkdir cache/torch_cache
```

2. Copy project
```bash
scp -P 14709 -r experiments llm2vec nlp_course scripts test_configs train_configs .env README.md setup.cfg setup.py root@ssh5.vast.ai:/root/llm2vec
```

3. Install project
```
scripts/install.sh
```

4. Copy fine-tuned model
```
scp -P 14709 -r output root@ssh5.vast.ai:/root/llm2vec
```

make sure model was copied successfully:

```bash
python scripts/sanity_check_trained_model.py --path ./output/mntp/dictalm2.0-instruct
```