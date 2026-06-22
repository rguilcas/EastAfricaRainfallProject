# BCCR Training Programme in Machine Learning
## Setup your ML workflow
---

This repository is an example machine learning project. 
In this README, you will find how to setup the project, and how to train a neural network, and make predictions. 


# SETUP 

## SETUP If running on your computer

This project uses a two-step setup:

1. Create a reproducible base environment from `extras/environment.yml`.
2. Install the platform-appropriate PyTorch backend (CUDA, MPS, or CPU).

### 1) Create or update base env

```bash
conda env create -f extras/environment.yml || conda env update -f extras/environment.yml --prune
```


### 2) Activate environment

```bash
conda activate bccr-ml-project
```

### 3) Install PyTorch backend

You can use the following bash script that will install pytorch with conda. 

```bash
bash extras/setup_pytorch_backend.sh --backend auto
```
You can also go to pytorch website and follow instructions to install pytorch with pip [](https://pytorch.org/get-started/locally/) 

<!-- 
Optional explicit backend selection:

```bash
bash extras/setup_pytorch_backend.sh --backend cuda --cuda-version 12.1
bash extras/setup_pytorch_backend.sh --backend mps
bash extras/setup_pytorch_backend.sh --backend cpu
``` -->

### 4) Setup weights and biases

Once the environment is activated, login to weights and biases to make sure experiments will get logged properly to your account.
You will need a weights and biases account. Then copy your API key and past it in the terminal when asked.

```bash
wandb login
```

## SETUP If running on HubroHub

### 1) Install missing libraries

If you are running this on HubroHub, you need to install lightning and wandb everytime you open a new session:

```bash
pip install wandb lightning
```

### 2) Setup weights and biases

Once all libraries are installed, login to weights and biases to make sure experiments will get logged properly to your account.
You will need a weights and biases account: [](https://wandb.ai/). Then copy your API key and past it in the terminal when asked.

```bash
wandb login
```



# Train the neural network and make predictions.

First navigate to the scripts/ directory:

```bash
cd scripts
```

Then, launch the train.sh script:

```bash
bash train.sh
```

You can specify a configuration file in config/ with 

```bash
bash train.sh --config config-file-name
```

Weights and biases will give a run ID, which you can reuse for the predictions.
The run ID can be found in the terminal output, e.g.

`wandb: 🚀 View run baseline at: https://wandb.ai/[wandb long user name]/bccr-ml-course/runs/`**RUNID**

```bash
bash predict.sh --run-id RUNID
```


