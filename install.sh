conda create -n maua
conda activate maua
conda install -y -c conda-forge cudatoolkit-dev
conda install -y -c pytorch pytorch torchvision
yes | pip install ffmpeg-python pyyaml scipy cupy

# git clone https://github.com/NVIDIA/apex
# cd apex/
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# rm -rf apex/

# cd ./correlation/
# rm -rf *_cuda.egg-info build dist __pycache__
# python setup.py install --user
