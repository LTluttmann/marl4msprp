# Multi-Agent Reinforcement Learning Algorithms for the Min-Max Mixed-Shelves Picker Routing Problem

Repository to solve the min-max Mixed-Shelves Picker Routing Problem (MSPRP) via multi-agent RL. This repository implements different network architectures (available in `marlprp/models`) and training algorithms (available in `marlprp/algorithms`). 



## Install
To run this code, first create a fresh environment and install the dependencies specified in the `requirements.txt`, e.g. using conda:
```
conda create --name marl4msprp --file requirements.txt
```

Then, install this library as a python package by running 

```
pip install -e .
```
in the same directory as the setup.py file




## Train 

This repository uses Hydra for experiment configuration. The configuration files can be found in `/configs/` folder and are strucured accoring to the different parts in the training pipeline:


##### Configuration Structure

```

config
│ 
│   main.yaml (The main configuration, composed of all sub-configs)
│   
└───env (defines the different environments)
│
└───policy (defines the different neural baseline policies)
│
└───model (defines the training algorithms)
│
└───env_list (defines a list of environments used for training and testing)
│
└───callbacks/hydra/logger/paths (define different loggers)
```

The configuration files can be adjusted when running the training script via the command line:

#### Training Run

```
python train.py env=25s-15i-50p model=sl policy=maham ++train.epochs=10 ++train.devices=[0]
```

Here we train the MAHAM policy on environments of type `25s-15i-50p` and use self-improvement (sl) as training algorithm. Further, we overwrite the default number of training epochs to 10 and use the GPU with index 0 as training device. Note, that GPU training is highly recommended. 

During training, metrics are logged using wandb. This requires the user to log-in via `wandb init`. Logging via wandb can be disabled by passing `logger=debug` via the CLI.
```
python train.py env=25s-15i-50p model=sl policy=maham logger=debug
```

## Test

Pre-trained models can be tested, for example on the datasets provided in the `data_test/large` folder, by using the `test.py` script. This script expects the `.hydra` path of the respective training run to be specified via the `--config-path` flag, because this enables us to load all the configurations from the training run.  
```
python test.py --config-path <ROOT>/logs/<RUN_ID>/.hydra --config-name=config
```
Note, that by default, this uses the environment configuration used during training for evaluating the model. To test out-of-distribution generalization, one can pass a different environment using the `test_env` flag:
```
python test.py --config-path <ROOT>/logs/<RUN_ID>/.hydra --config-name=config ++test_env=50s-500i-1000p
```



## Acknowledgements

This repository includes adaptions of the following repositories:
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/yd-kwon/MatNet
* https://github.com/ai4co/rl4co
* https://github.com/ai4co/parco