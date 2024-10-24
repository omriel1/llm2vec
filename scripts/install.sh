
# install vim
sudo apt install vim -y

# setup environment (conda is assumed)
pip install -e .
conda install -c conda-forge cudatoolkit-dev cuda-python
pip install flash-attn --no-build-isolation
