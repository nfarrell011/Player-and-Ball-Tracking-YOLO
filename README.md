# README
___ 

## Ball & Player Tracking - Team Differentiation

  CS 5330 Pattern Recognition & Computer Vision
  Professor Bruce Maxwell, PhD.
  Spring 2024
___

### Group Member Names:
* Joseph Nelson Farrell 
* Harshil Bhojwani
* Leonardo DeCraca
* Priyanka Dipak Gujar

___

### Links/Urls:

For links to image data, please the ```project_5_report.pdf```
___

### Operating System & IDE:
* MacOS
* Visual Studio Code

___
## Executing the Program:

### Step 1: Train Model

The first step is train the model on the digits dataset. To do so execute this command:
```bash
python3 task_1_train_model.py
```
A ```fig``` directory will be created if one does not exist. All figures of examples and results (for this step and all subsequent steps) can be viewed in the ```figs``` folder.

### Step 2: Test Model on a Sample from MNIST

To test the model on a sample of the first $10$ digits in MNIST execute this command
```
python3 task_1_execute_model_first_ten.py
```

### Step 3: Test Model of Homebrew Digit Images
To test the model on a set of hand drawn digits execute this command:
```
python3 task_1_execute_model_homebrew_digits.py
```

### Step 4: Examine Model
To examine the model, and specifically the first layer execute this command:
```
python3 task_2_examine_network.py
```
Along with the figures produced and placed in ```figs```, this will print to the terminal (partial example):
```
Model: 

NeuralNetwork(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
)

---------------------------------------------------------------------------
Filter: 0

Shape: (5, 5)

Weights:

[[ 0.19433641  0.2745412   0.06877322  0.3040398  -0.03765988]
 [-0.06402504  0.03289581  0.38099024  0.275251   -0.23104045]
 [ 0.12435468  0.18495612  0.3837142   0.09324145  0.07521386]
 [-0.05413244  0.20655519  0.1546273   0.03061795  0.11926898]
 [-0.16664755 -0.03156132  0.0316309   0.29251063 -0.07670494]]

---------------------------------------------------------------------------
Filter: 1

Shape: (5, 5)

Weights:

[[-0.15331498 -0.1533795  -0.20383942 -0.02902526 -0.25952595]
 [ 0.15782721 -0.19269173  0.0937702  -0.05408049 -0.17690542]
 [ 0.18173487  0.1222562   0.2812571   0.11163912 -0.05252617]
 [ 0.1236348   0.07647416  0.32179785  0.37122196  0.20810632]
 [-0.13126312  0.17672642  0.12961812  0.22738902  0.0409585 ]]

---------------------------------------------------------------------------
Filter: 2

Shape: (5, 5)

Weights:

[[-0.2031977   0.07933211  0.09610061  0.24562135  0.0173081 ]
 [ 0.07372052  0.28035986  0.23885728  0.14635111 -0.29419953]
 [ 0.17822446  0.26508206  0.29394704 -0.17245533 -0.19073011]
 [ 0.153382    0.4502485  -0.04520241 -0.2797144  -0.3472653 ]
 [ 0.30842462  0.21917397  0.12611137 -0.39402682 -0.47844738]]

-------------
```

### Step 5: Transfer Learning to Greek Letters
To transfer the model and train on Greek letters execute this command:
```bash
python3 task_3_greek_letter_train.py
```
This will display the modified model.
```
NeuralNetwork(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=3, bias=True)
)
```
### Step 6: Design Experiment: Gridsearch
To execute a gridsearch over 3 models and 2 hyperparameters execute this command. This can take a long time depending on ```epochs```. You can set this value as command line argument $2$. The default is $10$
```
python3 task_4_design_experiment.py
```

### Step 7: Extension: More Greeks
To train the model on more Greek letters, execute this command:
```
python3 task_5_extension_more_greeks.py
```



