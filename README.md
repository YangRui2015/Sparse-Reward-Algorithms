# Sparse-Reward-Algorithms

## Introduction

We implemented many classes of Sparse Reward algorithms in Gym Fetch environment including Reward Shaping, Imitation Learning, Curriculum Learning, Hindsight Experience Replay, Curiosity-Driven Exploration, Hierachical Reinforcement Learning. This work is for better understanding of sparse reward algorithms.

Our code is based on https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC- and we have changed a lot on code simplification and content richness.

## Usage

1. DDPG: 

``` python main.py --retrain ```

2. Reward Shaping:

``` python main.py --retrain --rtype dense```

3. Curriculum Learning:

``` python main.py --retrain --curriculum 2```

4. Imitation Learning:

``` python main.py --retrain --imitation --imit_ratio 1```

5. Hindsight Experience Replay:

``` python main.py --retrain --her```

6. Forward Dynamic:

``` python main.py --retrain --curiosity```

7. Hierachical DDPG:

``` python main.py --retrain --layers 2```

8. Test the latest saved checkpoint:

``` python main.py --test```

if using HDDPG, you should use :

``` python main.py --test --layers 2```

9. Save demostrations for imitation learning：

``` python main.py --retrain --her --save_experience```

## Result


![image](https://github.com/YangRui2015/Sparse-Reward-Algorithms/blob/master/data/result.jpg)




