# Risk-Aware Model-Based Control
A flexible risk-aware model-based Reinforcement Learning framework. This repository is the official implementation of under-review paper [Risk-Aware Model-Based Control]. 

## Requirements
* Python 3, Tensorflow 2
* To install requirements:
```setup
pip install -r requirements.txt
```
## Basic Usage
* Run RAMCO on the AntX model
```setup
cd src
python main.py --config antx --num_warmup 20000 --warmup_type 0 
```
>  Options for warmup_type: [0] Random, [1] SAC, [2] PPO
* Run RAMCO on the Eidos environment with 10-dimension state space
```setup
cd src
python main.py --config eidos --num_warmup 60000 --warmup_type 0 
```
* Run RAMCO on the Eidos environment with 100-dimension state space
```setup
cd src
python main.py --config eidose2 --num_warmup 60000 --warmup_type 0 
```
* Run RAMCO on the Eidos environment with 1000-dimension state space
```setup
cd src
python main.py --config eidose3 --num_warmup 60000 --warmup_type 0 
```
## Acknowledgment
** Parts of this repository are based on [nn_dynamics](https://github.com/nagaban2/nn_dynamics), [tf2rl](https://github.com/keiohta/tf2rl), and [gym](https://github.com/openai/gym). **
