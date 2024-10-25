# Connect
1. Copy the ssh command inorder to connect to the machine. Should look like
```
ssh -p {port} {server} -L 8080:localhost:8080
```

## First time setup

1. Create folders
```
mkdir llm2vec
mkdir llm2vec/output
mkdir cache
mkdir cache/hf_cache
mkdir cache/transformers_cache
mkdir cache/hf_dataset_cache
mkdir cache/torch_cache
```

2. Copy project
```bash
scp -P {port} -r experiments llm2vec nlp_course scripts test_configs train_configs .env README.md setup.cfg setup.py {server}:/root/llm2vec
```

3. Install project
Navigate to `llm2vec` and run:
```
scripts/install.sh
```

4. Copy fine-tuned model(s)

MNTP model:
```
scp -P {port} -r output/mntp {server}:/root/llm2vec/output
```

SimCSE model:
```
scp -P {port} -r output/mntp-simcse {server}:/root/llm2vec/output
```

make sure model was copied successfully:

```bash
python scripts/sanity_check_trained_model.py --path ./output/mntp/dictalm2.0-instruct
python scripts/sanity_check_trained_model.py --path ./output/mntp-simcse/dictalm2.0-instruct/checkpoint-800
```
