# RL-for-ramp-metring-on-highways

The goal of this project is to utilize Q-learning and Deep Q-learning algorithms through numerical simulation to optimize ramp metering control on a highway. The simulation focuses on a specific highway stretch with a configurable number of lanes, where an entry ramp is regulated by a traffic light. SUMO (Simulation of Urban Mobility) is employed to simulate car-following and lane-changing behaviors. The Q-learning algorithm is tasked with optimizing traffic flow on both the highway stretch and the ramp by controlling the ramp's traffic light.

# Requirements
    - SUMO
    - Traci
    - Pickle
    - Numpy
    - Tensorflow


# Models 
    - Q-Learning
    - Deep Q-Learning

# How to run

To visualize the simulation, run the following command in the terminal:
```python visualize_dql.py``` or ```python visualize_ql.py``` 

To train the model, run the following command in the terminal:
```python train_dql.py``` or ```python train_ql.py```

To test the model, run the following command in the terminal:
```python test_dql.py``` or ```python test_ql.py```


