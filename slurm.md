# Connect
1. Connect to TAU vpn
2. Connect to the login server (used to submit jobs to the worker nodes)
Start with:
```
ssh <user-name>@c-002.cs.tau.ac.il
```
And enter your password.

3. Navigate to your course folder
```
cd /home/joberant/NLP_2324b/<user-name>
```

## Not the first time?
Initialize the conda environment:
```
bash
conda activate llm2vec
```

And start working!

## First time setup
**Just on first time**
1. Create cache folders (**Just on first time**)
```
mkdir cache
mkdir llm2vec
mkdir cache/hf_cache
mkdir cache/transformers_cache
mkdir cache/hf_dataset_cache
mkdir cache/torch_cache
```

2. Install Anaconda and create a virtual environment
```
wget repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
```
This will download a script called `Anaconda3-2020.11-Linux-x86_64.sh`.
Now run it, and press `enter` until you need to accept the license by typing "yes"
```
sh Anaconda3-2020.11-Linux-x86_64.sh 
```
**Do not** user the default location, instead:
```
/home/joberant/NLP_2324b/<user-name>/anaconda3
```

After (the **long**) installation, when you're asked: 
`Do you wish the installer to initialize Anaconda3
by running conda init?` just enter `yes`.

3. Create virtual environment
First, use `bash` to activate the base virtual environment:
```
bash
```
And create a new virtual environment:
```
conda create -n llm2vec
```
And now you can activate it using:
```
conda activate llm2vec
```

4. Copy the project into slurm

The general command to copy files is: 
```
rsync -vrah <path-on-local-machine> <user-name>@c-002.cs.tau.ac.il:<path-on-slurm> --stats --progress
```

Now, lets copy the entire project **excluding** our virtual env. Make sure you run this command **on your local machine**
from the projects' base directory.
```
rsync -vrah experiments llm2vec nlp_course scripts test_configs train_configs .env README.md setup.cfg setup.py <user-name>@c-002.cs.tau.ac.il:/home/joberant/NLP_2324b/<user-name>/llm2vec --stats --progress
```

5. install required packages
First make sure you're using the created virtual environment (llm2vec). If not:
```
bash
conda activate llm2vec
```

Navigate to the `llm2vec` folder and run:
```
./scripts/install.sh
```
