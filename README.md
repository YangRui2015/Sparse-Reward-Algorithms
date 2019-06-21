# HDDPG + HER + RND
This repository contains the code to implement the *Hierarchical Deep Deterministic Policy Gradient (HDDPG) & Hindsight Experience Replay(HER) & Random Network Distillation(RND)* algorithm. Our experiment environment is Mocojo Robot environment, including *Reach、Push、PickandPlace、Slide*. However, We only finished the Reach task till now.

To run the codes, you can first execute the command *"python run_HAC.py --layers 1 --her --normalize  --retrain  --env reach  --episodes 5000 --threadings 1"*. The meaning of the flag is easy to understand, and you can read the option.py file to see all the flags. There is a "performance.jpg" showing the accuracy of training only if the threadings is 1.

Our RND is an off-policy implement as most of the popular Curiosity Driven methods are on-policy recently, so we need to compute the intrinsic reward every batch sampled from the replay buffer because it changes when training.

More details will be added later.

Thanks to the author of HAC, HER and RND. 

## Version LOG

### 2019/5/7 First Version
1.  Hierachical DDPG and HER;

2.  Observation (State/Goal) Normalization;

3.  RND;

4.  Mutilprocessing (so we can run many experiments in the same time);

5. Reach and Push environment;

### 2019/5/10 Update
1. Use gym to create environment class(so it is easy to use other environment);

2. Hand Reach environment;
