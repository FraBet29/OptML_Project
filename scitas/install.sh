# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
conda config --set pip_interop_enabled True
conda update --all && conda clean -p
conda config --set auto_activate_base false

# Install Python
ENV_NAME=dl
conda create -n ${ENV_NAME} python=3.10
conda activate ${ENV_NAME}
