# Troubleshooting Guide — Common Issues and Solutions

Q1. Issues with environment configuration

Problem: If I don't use Singularity, how should I prepare the environment?

Solution: Please follow these steps.

- Create a Conda environment.
    ```shell
    conda create -n colongpt python=3.10
    conda activate colongpt
    ```

- Upgrade pip and install basic libraries:
    ```shell
    pip install --upgrade pip
    pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
    ```

- To install [NVIDIA Apex](https://github.com/NVIDIA/apex) for optimized speed with mixed precision training, you must compile the library manually. 
    ```shell
    pip install ninja
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    ```

- Install [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), which provide a fast and memory-efficient exact attention for optimised training.
    ```shell
    pip install packaging
    pip install flash-attn==2.7.0.post2 --no-build-isolation
    ```

- Please clone our repository and install the above dependencies. If `git` cannot be used within the singularity/docker container, kindly exit the container environment and perform the operation outside.
    ```shell
    git clone git@github.com:ai4colonoscopy/IntelliScope.git
    cd IntelliScope
    pip install script/requirements.txt
    pip install -e .
    ```