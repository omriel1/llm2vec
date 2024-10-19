pip install -e .

# install flash-attention related
# pip install packaging
# pip install ninja
# conda install -c nvidia cuda-python
conda install -c conda-forge cudatoolkit-dev cuda-python
pip install flash-attn --no-build-isolation
