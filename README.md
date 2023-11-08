# Robust-DPFL
The codes of our Robust-DPFL method. 

## Code Files
**preprocessing.py:** Functions for loading and processing the raw data of different experimental datasets, including MNIST, FEMNIST, and CIFAR10.

**poison.py:** Functions for posioning the data via backdoor attacks.

**model.py:** Functions of the deep learning model trained in experiments.

**FLTrain.py:** Functions of the federated learning workflow, including different attack strategies on DP-FL, and our Robust-DPFL method.

**Main.py:** Functions of the training and evaluation workflow.

## Data Files
```
Data
│---MNIST: Training and test data of MNIST 
│---CIFAR10: Training and test data of CIFAR10
```

## Data Files
```
Result
│--- Files for saving experimental results.
```

## Command 
#### Parameters
**-d**: *--dataset*, parameter for controlling the expriment datasets, legal values: *MNIST*, *FEMNIST*, *CIFAR10*

**-a**: *--attack-mode*, parameter for controlling the attack strategy, legal values: *AttackNaive*, *AttackNonDP*, *AttackDPFL*

**-m**: *--defense-mode*, parameter for controlling the federated gradient aggergation strategy, legal values: *FedAvg*, *RobustDPFL*

**-t**: *--taxic-ratio*, parameter for controlling the ratio of malicious client, legal value: a float number ranging from 0 to 1

**-a**: *--alpha*, paramter for controlling the privacy levels, legal value: a float number greater than 1

**-e**: *--epsilon*, paramter for controlling the privacy levels, legal value: a float number greater than 0

**-g**: *--gpu*, paramter for controlling the used GPU ID, legal value: a int number

#### Quick Command
python Main.py -d MNIST -t 0.15 -a AttackDPFL -m RobustDPFL