# Setup
1. Clone the project locally and make sure you use on the `development` branch, and it's up-to-date.

2. Make sure you have the fine-tuned models (`mntp` and `mntp-simcse`) locally 
If not, please download it from: https://drive.google.com/drive/folders/1u5ogHlfpeM84nY-LoqjGjRRx94Okq-6o?usp=drive_link
You should download the `output` folder as-is, and locate it in the projects' top directory.

# Vast.ai
Assuming you've rented a machine and you want to use/test the fine-tuned models, or perform
a fine-tuning from scratch:

1. Copy the ssh command inorder to connect to the machine. Should look like
```
ssh -p {port} {server} -L 8080:localhost:8080
```

2. Create the following folders:
```
mkdir llm2vec
mkdir llm2vec/output
mkdir cache
mkdir cache/hf_cache
mkdir cache/transformers_cache
mkdir cache/hf_dataset_cache
mkdir cache/torch_cache
```

3. Copy the project into the new machine
```bash
scp -P {port} -r experiments llm2vec nlp_course scripts test_configs train_configs .env README.md setup.cfg setup.py {server}:/root/llm2vec
```

4. Setup the development environment
Navigate to `llm2vec` and run:
```
scripts/install.sh
```

5. If desired, copy the fine-tuned models to the new machine:

MNTP model:
```bash
scp -P {port} -r output/mntp {server}:/root/llm2vec/output
```

SimCSE model:
```bash
scp -P {port} -r output/mntp-simcse {server}:/root/llm2vec/output
```

6. Make sure model was copied successfully:

```bash
python scripts/sanity_check_trained_model.py --path ./output/mntp/dictalm2.0-instruct
python scripts/sanity_check_trained_model.py --path ./output/mntp-simcse/dictalm2.0-instruct/checkpoint-1000
```

