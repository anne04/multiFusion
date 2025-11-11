# Python package installation

This project is developed using Python 3.10.13 and Pytorch 2.2.2 (with CUDA 12.1). After installing Python we can create a virtual environment and activate it before installing other Python libraries to setup the environment:
```
python -m venv multifusion_venv

source multifusion_venv/bin/activate
```
Upgrade pip:
```
pip install --no-index --upgrade pip
```

After that, use the requirements.txt to install the required Python libraries as follows:
```
pip install -r requirements.txt
```

The above command should setup the environment properly. If the Pytorch setup fails due to unmatched CUDA in your system, 
then please install it manualy based on your Pytorch and CUDA version (https://pytorch.org/get-started/previous-versions/). 
Here we are using pytorch 1.13.1 with CUDA 11.7:

```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip3 install torch-geometric -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

After the environment is setup, you can exit from your virtual environment using following command:
```
deactivate
```

Please remember to activate the 'multifusion_venv' using 'source' command everytime you run multiFusion model. 





