### All old literature review is here
![[highlights.pdf]]



#### Paper Number 77
#### Title 
#### One-liner summary
#### Summary

#### Paper Number 76
#### Adaptive Risk Minimization: Learning to Adapt to Domain Shift #RL/ModelBased #Meta-Learning  #Adaptation 
[PDF from Semantic Scholar](https://www.semanticscholar.org/reader/58a4a8e23256e0c9dd5071de0587b84f5f88d8c9)

#### One-liner summary
*"Our main contribution is to introduce the framework of adaptive risk minimization (ARM), which  
proposes the following objective: optimize the model such that it can maximally leverage the  
unlabeled adaptation phase to handle domain shift. To do so, we instantiate a set of methods that,  
given a set of training domains, meta-learns a model that is adaptable to these domains"*
#### Summary

#### Paper Number 75
#### Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization #RL/ModelBased 

[Annotated PDF in Semantic Scholar](https://www.semanticscholar.org/reader/79ebde314ab90d066cee3b82193ef05666323394)

#### One-liner summary
*"To achieve high deployment efficiency, we propose Behavior-Regularized Model-ENsemble (BRE-
MEN). BREMEN incorporates Dyna-style [58 ] model-based RL, learning an ensemble of dynamics
models in conjunction with a policy using imaginary rollouts from the ensemble and behavior
regularization via conservative trust-region updates."*

==-> Policy again learnt in the model of the model based learning==
#### Summary
![[Pasted image 20240104175534.png]]
#### Paper Number 74
#### DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION #RL/ModelBased #World-Model  
[Annotated PDF on Semantic Scholar](https://www.semanticscholar.org/reader/0cc956565c7d249d4197eeb1dbab6523c648b2c9
)
#### One-liner summary
*"We present Dreamer, a reinforcement learning  
agent that solves long-horizon tasks from images purely by latent imagination.  
We efficiently learn behaviors by propagating analytic gradients of learned state  
values back through trajectories imagined in the compact state space of a learned  
world model."*

==-> Key point here is that they also have a model for the value estimates that they learn using analytical gradients of their world model (this value model comes from an Actor Critic method)==
==-> Again note that the action model and the value model is learnt in the latent space of the world model==
==-> Only because their world model is "differentiable" (because its just neural networks), they can back propogate to train the value network==
#### Summary
1) #Method What the paper brings to the table 
	1) *"The key contributions of this paper are summarized as follows:  
• **Learning long-horizon behaviors by latent imagination**: Model-based agents can be short-  
sighted if they use a finite imagination horizon. We approach this limitation by predicting both  
actions and state values. Training purely by imagination in a latent space lets us efficiently learn  
the policy by propagating analytic value gradients back through the latent dynamics.  
• **Empirical performance for visual control** : We pair Dreamer with existing representation  
learning methods and evaluate it on the DeepMind Control Suite with image inputs, illustrated in  
Figure 2. Using the same hyper parameters for all tasks, Dreamer exceeds previous model-based  
and model-free agents in terms of data-efficiency, computation time, and final performance"*
2) #Method Things that are learnt
	1) Latent Dynamics model (representation model, transition model and reward model) from past experience to predict future rewards from actions and past observations
	2) Action and value models from predicted latent trajectories. The value model optimizes Bellman consistency for imagined rewards and the action model is updated by propagating gradients of value estimates back through the neural network dynamics
3) #Method Action and value models use latent states
	1) *"We learn an action model and a value model in the latent space of the world model for this. The action model implements the policy and aims to predict actions that solve the imagination environment"*
4) #Citable Why learning the reward model directly is not great
	1) *"In principle, this could be achieved by simply learning to predict future rewards given actions and past observations (Oh et al., 2017; Gelada et al., 2019; Schrittwieser et al., 2019). With a large and diverse dataset, such representations should be sufficient for solving a control task. However, with a finite dataset and especially when rewards are sparse, learning about observations that correlate with rewards is likely to improve the world model"*
Did not understand very well how these networks are trained.
#### Paper Number 73 
#### Adaptive Online Planning for Continual Lifelong  Learning #RL/Hybrid 
[Annotated PDF on Semantic Scholar](https://www.semanticscholar.org/reader/1690d6db86ed751bf7fb29fb768d1418ba579abc)

#### One-liner summary
*"We present a new algorithm,  
Adaptive Online Planning (AOP), that is capable of achieving strong performance  
in this setting by combining model-based planning with model-free learning. By  
measuring the performance of the planner and the uncertainty of the model-free  
components, AOP is able to call upon more extensive planning only when necessary,"*

-> **Uses an ensemble of value functions that go into the reward for MPC**
#### Summary
1) #Citable Model based vs Model free
	1) *"Model-based trajectory optimization via planning is useful for quickly learning control, but is  
computationally demanding and can lead to bias due to the finite planning horizon. Model-free  
reinforcement learning is sample inefficient, but capable of cheaply accessing past experience without  
sacrifices to asymptotic performance"*

2) #Method How is using a Planner vs Learned policy solved here? 
	1) *"Deciding when to use the planner vs a learned policy presents a difficult challenge, as it is hard to  determine the improvement the planner would yield without actually running the planner. We tackle this as a problem of uncertainty. When uncertain about a course of action, humans use an elongated model-based search to evaluate long-term trajectories, but fall back on habitual behaviors learned with model-free paradigms when they are certain of what to do"*
	2) The model has access to ground truth dynamics. This is a #Details that should not be missed
3) #Method The different algorithms used
	1) *"We present a new algorithm, Adaptive Online Planning (AOP), that links Model Predictive Path Integral control (MPPI) [32], a model-based planner, with Twin Delayed DDPG (TD3) [ 12 ], a model-free policy learning method"*
	2) *"We combine the model-based planning method of iteratively updating a planned trajectory with the model-free method of updating the network weights to develop a unified update rule formulation that is amenable to reduced computation when combined with a switching mechanism. We inform this mechanism with the uncertainty given by an ensemble of value functions."*
4) #Method Early Planning Termination
	1) They use early planning termination (MPC does not run for fixed number of iterations but for iterations till there is an improvement above a certain threshold). *"When this improvement decreases below a threshold ∆thres, we terminate planning for the current timestep with probability 1 − plan. Using a stochastic termination rule allows for robustness against local minima where more extensive planning may be required, but not evident from early planning iterations, in order to escape."*
5) #Method Adaptive Planning Horizon
	1) #Details on Page 4
6) #Method Model Free TD3 as a prior to the planning procedure
	1) *"We use TD3 as a prior to the planning procedure, with the policy learning off of the data generated by the planner during planning, which allows the agent to recall past experience quickly"*
7) 


#### Paper Number 72
#### When to Trust Your Model: Model-Based Policy Optimization #RL/ModelBased 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/9001698e033524864d4d45f051a5ba362d4afd9e)
#### One-liner summary
**ChatGPT:** *This method efficiently balances the use of model-generated data and real data, mitigating the bias inherent in model-generated data. The approach demonstrates superior sample efficiency and asymptotic performance compared to existing model-based methods, while avoiding pitfalls such as model exploitation. The paper provides both theoretical analysis and empirical evidence to support the effectiveness of using short model-generated rollouts in reinforcement learning.*
#### Summary
1) They seem to judiciously use the learnt model and the actual simulator in gaining data to train their RL policy
2) #Citable 
	1) *"However, an empirical study of  model generalization shows that predictive models can indeed perform well outside of their training distribution."*

#### Paper Number 71
#### MODEL-ENSEMBLE TRUST-REGION POLICY OPTIMIZATION #RL/ModelBased #ProbabilisticDynamicModels 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/27dfecb6bb0308c7484e13dcaefd5eeebba677d3)

#### One-liner summary
**Paper that shows that an ensemble of learnt models acts as a regularizer**
#### Summary
1) #Citable 
	1) *"Model uncertainty is a principled way to reduce model bias"*

#### Paper Number 70
![[2021ArgensonModelBasedOfflinePlanning.pdf]]
#### MODEL-BASED OFFLINE PLANNING - #RL/ModelBased #ProbabilisticDynamicModels #StochasticPlanning 

#### One-liner summary
*"Our proposed algorithm, MBOP (Model-Based Offline Planning), is a model-based RL algorithm
able to produce performant policies entirely from logs of a less-performant policy, without ever
interacting with the actual environment. MBOP learns a world model and leverages a particle-based
trajectory optimizer and model-predictive control (MPC) to produce a control action conditioned on
the current state."*

**Its a combination of Paper 69 and 68**
#### Summary
1) #Citable More advantages of Model Based RL
	1) *"This is interesting because the final policy can be more easily adapted to new tasks, be made to respect constraints, or offer some level of explainability. When bringing learned controllers to industrial systems, many of these aspects are highly desirable, even to the expense of raw performance."*
2) #Citable Adaptive nature of model based RL
	1) *"We show that MBOP successfully integrates constraints that were not initially in the dataset and is able to perform well on objectives that are different from the objective of the behavior policy"*
3) #Citable Conclusions
	1) *"MBOP provides an easy to implement, data-efficient, stable, and flexible algorithm for policy generation. It is easy to implement because the learning components are simple supervised learners, it is data-efficient thanks to its use of multiple complementary estimators, and it is flexible due to its use of on-line planning which allows it to dynamically react to changing goals, costs and environmental constraints."*
4) #Method The NN's involved
![[Pasted image 20240103112230.png]]
3) #Method** Extension of Paper Number 69**
	1) *"MBOP-Trajopt extends ideas used by PDDM (Nagabandi et al., 2020) by adding a policy prior (provided by fb ) and value prediction (provided by fR )."*
	2) In  essence they do iterative guided-shooting trajectory optimization with refinement
	3) #Details of algorithm in page 5
#### Paper Number 69
####  Deep Dynamics Models for Learning Dexterous Manipulation #RL/ModelBased #ProbabilisticDynamicModels  #StochasticPlanning
[Annotated PDF from Semantic Scholar](https://www.semanticscholar.org/reader/7a450675968d31b8363e21fb5d5b72474c128076)

#### One-liner summary
Online Planning with Deep Dynamics Models (PDDM)
*"we show that improvements in learned dynamics models, together with improvements in online model-predictive control, can indeed enable efficient and effective learning of flexible contact-rich dexterous manipulation skills – and that too, on a 24-DoF anthropomorphic hand in the real world, using just 4 hours of purely real-world data to learn to simultaneously coordinate multiple free-floating objects"*

#### Summary
1) #Citable Complex physics needed for dexterous manipulation
	1) *"The principle challenges in dexterous manipulation stem from the need to coordinate numerous joints and impart complex forces onto the object of interest. The need to repeatedly establish and break contacts presents an especially difficult problem for analytic approaches, which require accurate models of the physics of the system."*
2) #Citable One of the first applications of using model based RL for complex tasks
	1) *"Our approach, based on deep model-based RL, challenges the general machine learning community’s notion that models are difficult to learn and do not yet deliver control results that are as impressive as model free methods"*
	2) 
3) #Citable Probabilistic models are good
	1) *"As prior work has indicated, capturing epistemic uncertainty in the network weights is indeed important in model-based RL, especially with high-capacity models that are liable to overfit to the training set and extrapolate erroneously outside of it"*
	2) #Definition **Bootstrap ensembles** - *"approximate the posterior p(θ|D) with a set of E models, each with parameters θi. For deep models, prior work has observed that bootstrap resampling is unnecessary, and it is sufficient to simply initialize each model θi with a different random initialization $\theta_0^i$ and use different batches of data $D_i$ at each train step"  
4) #Citable Why Model-based RL might be more data efficient
	1) *"We note that this supervised learning setup makes more efficient use of the data than the counterpart model-free methods, since we get dense training signals from each state transition and we are able to use all data (even off-policy data) to make training progress."*
4) #Method This is a combination of many methods
	1) *"Our method combines components from multiple prior works, including uncertainty estimation deep models and model-predictive control (MPC), and stochastic optimization for planning"*
	2) They use online planning with MPC to select actions via  model predictions
	3) #Details **They have explanations for a bunch of optimizer they use in the MPC loop on page 4**
		1) Random shooting
		2) Iterative Random-shooting with Refinement
		3) Filtering and Reward-Weighted Refinement
5) #Results Ablation studies for the various choices they made
![[Pasted image 20240103093609.png]]
6) #Results Outperforms other model-based and model-free approaches
![[Pasted image 20240103093846.png]]
7) #Results Smashing
	1) *"As we show in our experiments, our method achieves substantially better results than prior deep model-based RL methods, and it also has advantages over model-free RL: it requires substantially less training data and results in a model that can be flexibly reused to perform a wide variety of user-specified tasks. In direct comparisons, our approach substantially outperforms state-of-the-art model-free RL methods on tasks that demand high flexibility, such as writing user-specified characters. In addition to analyzing the approach on our simulated suite of tasks using 1-2 hours worth of training data, we demonstrate PDDM on a rea-lworld 24 DoF anthropomorphic hand, showing successful in-hand manipulation of objects using just 4 hours worth of entirely real-world interactions"*
	

![[2019LowreyPOLO.pdf]]
#### Paper Number 68
#### Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control #RL/ModelBased 

#### One-liner summary
*"The POLO framework combines three components: local trajectory optimization, global value function approximation, and an uncertainty and reward aware exploration strategy."*

**This could be a great algo for sparse reward exploration problems**
Combines planning with learning of an approximate value function
#### Summary
1) #Citable The strengths of MPC
	1) *"MPC (with H > 1) is less susceptible to approximation errors than greedy action selection."*
	2) *"MPC can also enable faster convergence of the value function approximation"*
	3) *"The ability of an agent to explore the relevant parts of the state space is critical for the convergence of many RL algorithms. Typical exploration strategies like -greedy and Boltzmann take exploratory actions with some probability on a per time-step basis. Instead, by using MPC, the agent can explore in the space of trajectories. The agent can consider a hypothesis of potential reward regions in the state space, and then execute the optimal trajectory conditioned on this belief, resulting in a temporally coordinated sequence of actions. By executing such coordinated actions, the agent can cover the state space more rapidly and intentionally, and avoid back and forth wandering that can slow down the learning."*
2) #Method 
	1) They have an ensemble of value functions that they train using Maximum Likelihood
	2) #Details **Final algorithm in page 5**
3) #Method  In own words because the text is too convoluted
	1) They use a standard model as the M of MPC
	2) However they learn ensemble of value functions and use that for global exploration while using MPC for planning. This ensemble of value functions tracks the "uncertainty" associated with not visiting certain states




#### Paper Number 67
#### Gradient-based Planning with World Models #RL/ModelBased #World-Model  
[Annotated PDF from Semantic Scholar](https://www.semanticscholar.org/reader/5c9eafe8d21052095473516297c4b5a24373250c)

#### One-liner summary
Use gradient while planning
#### Summary
1) #Contributions
	1) **Gradient-Based MPC:** We employ gradient-based planning to train a world model based  on reconstruction techniques and conduct inference using this model. We compare and  contrast the performance of traditional population-based planning methods, policy-based  methods, and gradient-based MPC in a sample-efficient setting involving 100,000 steps in  the DeepMind Control Suite tasks. Our approach demonstrates superior performance on many tasks and remains competitive on others.
	2) **Policy + Gradient-Based MPC:** We integrate gradient-based planning with policy networks, outperforming both pure policy methods and other pure MPC techniques in sparse reward environments.
2) Paper is horribly written didn't get much more out of it

#### Paper Number 66
#### World Models #RL/ModelBased #World-Model

[Annotated PDF in Semantic Scholar](https://www.semanticscholar.org/reader/ff332c21562c87cab5891d495b7d0956f2d9228b)
#### One-liner summary
*"Our world model can be trained quickly in an unsupervised manner to learn a compressed spatial and temporal representation of the environment. By using features extracted from the world model as inputs to an agent, we can train a very compact and simple policy that can solve the required task. We can even train our agent entirely inside of its own hallucinated dream generated by its world model, and transfer this policy back into the actual environment."*
#### Summary
1) #Citable Why RL uses small networks
	1) *"The RL algorithm is often bottlenecked by the credit assignment problem, which makes it hard for traditional RL algorithms to learn millions of weights of a large model, hence in practice, smaller networks are used as they iterate faster to a good policy during training."*
2) #Citable Learning in a nightmare produces better performance in reality
	1) *"an agent that is able to survive the noisier and uncertain virtual nightmare environment will thrive in the original, cleaner environment."*
3) #Citable Exploiting nature of RL agents
	1) *"The weakness of this approach of learning a policy inside a learned dynamics model is that our agent can easily find an adversarial policy that can fool our dynamics model – it’ll find a policy that looks good under our dynamics model, but will fail in the actual environment, usually because it visits states where the model is wrong because they are away from the training distribution."*
4) #Citable Why training in a dream is useful
	1) *"We have demonstrated the possibility of training an agent to perform tasks entirely inside of its simulated latent space dream world. This approach offers many practical benefits. For instance, running computationally intensive game engines require using heavy compute resources for rendering the game states into image frames, or calculating physics not immediately relevant to the game. We may not want to waste cycles training an agent in the actual environment, but instead train the agent as many times as we want inside its simulated environment. Training agents in the real world is even more expensive, so world models that are trained incrementally to simulate reality may prove to be useful for transferring policies back to the real world."*
	2) 
5) #Method Training scheme
	1) *"We first train a large neural network to learn a model of the agent’s world in an unsupervised manner, and then train the smaller controller model to learn to perform a task using this world model"*
	2) Use **variational auto encoders** to learn latent space from image. Then use **RNNs** to predict next latent embedding to expect. The controller is then just  a **linear model** that takes this latent embedding and the hidden state of the RNN to prdouce the actions.  All are trained seperately
	![[Pasted image 20240102143046.png|300]]
	![[Pasted image 20240102144030.png|300]]
	
	3) Surprisingly, C is trained with CMA-ES on multiple CPU cores
3) #Method The agent is then also completely trained in the "dream" environment characterized by the latent space.

	
#### Paper Number 65
#### Meta Reinforcement Learning with Latent Variable Gaussian Processes #RL/ModelBased #ProbabilisticDynamicModels 
[Annotated PDF from Semantic Scholar](https://www.semanticscholar.org/reader/6d561e0d7187e308916cde746aae9b4aa6658d17)

#### One-liner summary
*"Hence, we systematically combine three orthogonal ideas (probabilistic models, MPC, meta learning) for increased data efficiency in settings where we need to solve different, but related task

#### Summary
1) #Citable Model based RL challenges
	1) *"A challenge with these learned models is the problem of model errors: If we learn a policy based on an incorrect model, the policy is unlikely to succeed on the real task. To mitigate the issue of these model errors it is recommended to use probabilistic models and to take model uncertainty explicitly into account during planning"*
2) #Citable Why probabilistic models help
	1) *"probabilistic models of f are essential for data-efficient learning as they mitigate the  effect of model errors."*
3) #Method
	1) *"Conditioning the GP on the latent variable enables it to disentangle global and task specific variation  in the dynamics. Generalization to new dynamics is  done by inferring the latent variable of that system"* - The GP here is the dynamics model
	2) **I dont understand very well how they train, might beed a revisit**
4) #Method - Summary of entire method
	1) *"The key idea behind our approach is to address the meta learning problem probabilistically  using a latent variable model. We use online variational  inference to obtain a posterior distribution over the latent  variable, which describes the relatedness of tasks. This  posterior is then used for long-term predictions of the  state evolution and controller learning within a modelbased RL setting."*
5) #Results 
	1) *"We demonstrated that our ML-GP approach is as efficient or better than a non-meta learning baseline when solving multiple tasks at once. The ML-GP further generalizes well to learning models and controllers for unseen tasks giving rise to substantial improvements in data-efficiency on novel tasks"*


#### Paper Number 64
#### LEARNING TO ADAPT IN DYNAMIC, REAL-WORLD  ENVIRONMENTS THROUGH META-REINFORCEMENT LEARNING #RL/ModelBased #Meta-Learning #Adaptation

[Annoted PDF in Semantic Scholar](https://www.semanticscholar.org/reader/944bd3b472c8a30163bbfc1b5cbab8545693c3e0)


#### One-liner summary
*"Our approach uses meta-learning to train a dynamics model prior such that, when combined with  
recent data, this prior can be rapidly adapted to the local context"*
Primary goal is to achieve online adaptation in dynamic environments.
Dynamics model is  Neural Network with gaussian noise. For algorithm see page 6.
The planner used is MPPI in sim and MPC with random shooting in reality.

#### Summary
1) #Citable 
	1) *"Learning to adapt a model alleviates a central challenge of model-based reinforcement learning: the  problem of acquiring a global model that is accurate throughout the entire state space. Furthermore,  even if it were practical to train a globally accurate dynamics model, the dynamics inherently change  as a function of uncontrollable and often unobservable environmental factors."*

2) Alot of good #Citable for meta-learning on Page 3 and 4
3) Also #Citable is another result that Model-based approaches are sometimes not very good in asymptotic performance.
4) The dynamics is $NN(\theta')$ where $\theta'$ is obtained from the update rule (meta-learning stuff) $\theta' = u_{\psi}(\tau, \theta)$ where $\tau$ is the data set of M time steps. Loss is the negative LL of the data under the dynamics model
	1) M points are used to adapt $\theta$ to $\theta'$. And the $\theta'$ is evaluated on the future K points
5) #Results 
	1) Having model adaptation improves errors compared to TRPO using domain randomization (Sec 6.1)
	2) Requires 1000 times less data than model-free (Sec 6.2)
	3) Their methods that are trained for adaptation perform better and adapt faster than models that were just trained and made to adapt at test time and also a model that was trained for a high number of time steps on the test environment itself (Sec 6.3).
	4) On the real robot, their method out performs Model Based methods and Model Based methods with Dynamic evaluation. *"1)adapt online to a missing leg, 2) adjust to novel terrains and slopes, 3) account  for miscalibration or errors in pose estimation, and 4) compensate for pulling payloads"* 

#### Paper Number 63
#### Deep Reinforcement Learning in a Handful of Trials  using Probabilistic Dynamics Models #RL/ModelBased #ProbabilisticDynamicModels #StochasticPlanning 
[Annotated PDF on Semantic Scholar](https://www.semanticscholar.org/reader/56136aa0b2c347cbcf3d50821f310c4253155026)

#### One-liner summary
*"We propose a new algorithm called probabilistic ensembles with trajectory sampling (PETS) that combines* uncertainty-aware deep network dynamics models with sampling-based uncertainty 
*propagation*"
They use an **ensemble of bootstrapped probabilistic neural networks as their dynamics models.**
#### Summary
1) #Citable - MBRL vs Model Free
	 "*However, the asymptotic performance of MBRL methods  on common benchmark tasks generally lags behind model-free methods. That is, although MBRL  methods tend to learn more quickly, they also tend to converge to less optimal solutions.* "
 2) #Citable - Gaussian process drawbacks for dynamics modelling
	 *"while efficient models such as Gaussian processes can learn extremely quickly,  they struggle to represent very complex and discontinuous dynamical systems.By contrast, neural network (NN) models can scale to large datasets with high-dimensional inputs, and can represent such systems more effectively. However, NNs struggle with the opposite problem: to learn fast means to learn with few data and NNs tend to overfit on small datasets, making poor predictions far into the future.*"
 3) #Citable - Why uncertainty in Dynamics is good
	 *"Our second observation is that this issue can, to a large extent, be mitigated by properly incorporating uncertainty into the dynamics model"*
 4) #Method  once the dynamics is learnt
    "*Once a dynamics model ̃f is learned, we use ̃f to predict the distribution over state-trajectories*  
	*resulting from applying a sequence of actions. By computing the expected reward over state-*  
	*trajectories, we can evaluate multiple candidate action sequences, and select the optimal action*  
	*sequence to us*"
5) #Citable - Model choice in MBRL is crucial
	*"Any MBRL algorithm must select a class of model to predict the dynamics. This choice is often crucial  for an MBRL algorithm, as even small bias can significantly influence the quality of the corresponding  controller*"
6) #Citable - Importance of distinguishing types of uncertainty
	*"Without a  way to distinguish epistemic uncertainty from aleatoric, an exploration algorithm (e.g. Bayesian  optimization) might mistakingly choose actions with high predicted reward-variance ‘hoping  to learn something’ when in fact such variance is caused by persistent and irreducible system  stochasticity offering zero exploration value."*
	
7) #Results - Probabilistic Ensemble models win
	"*the probabilistic ensembles (PE-XX) perform best in*  
	*all tasks, except cartpole (‘X’ symbolizes any character). Close seconds are the single-probability type models: probabilistic network (P-XX) and ensembles of deterministic networks (E-XX). Worst* is the deterministic network (D-E).*"
	"Our results indicate that the gap in asymptotic performance between model-based and model-free  reinforcement learning can, at least in part, be bridged by incorporating uncertainty estimation into  the model learning process"
 
 
 
#### Paper Number 62
[Annoted PDF in Semantic Scholar](https://www.semanticscholar.org/reader/1df5d8dbc02ff5d8489a2c1d4514eeef56188b39)

#### PIPPS: Flexible Model-Based Policy Search  Robust to the Curse of Chaos #RL/ModelBased #ProbabilisticDynamicModels

#### One-liner summary
Provides an improvement on PILCO by using a better and more stable gradient estimate of the model. The algorithm is called **PIPPS**

#### Summary
1) They say that the problem of exploding gradients may be caused by the fundamental chaos-like nature of long chains of nonlinear computations with the magnitude becoming large and the direction becoming essentially random - ***"Thus, this work searches for an alternative flexible***  
***method for evaluating trajectory distributions and gradients"***
2) They show however that likelihood gradients don't suffer much from this problem than the reparametrization gradients do (this is related to differentiating through stochasticties)
3) They introduce a total propagation algorithm for getting a gradient from a population
4) The large variance in the gradients when using particles with reparameterization is due to the chaos-like sensitive dependence on the initial conditions - a common property in long chains of nonlinear mappings
5) Section 5.2: The Curse of Chaos in Deep Learning is a really good paragraph to understand the exploding gradients issue

#### Paper Number 61
[Link to Semantic Scholar](https://www.semanticscholar.org/reader/ab68bd6f47bfa8744f0f39be8c163d28203eefa2)

#### Bayesian Optimization with Automatic Prior Selection  for Data-Efficient Direct Policy Search #RL/ModelBased 
#### One-liner summary
Bayesian optimization is again used to model the Long term reward function. The contribution is a new acquisition function to choose the next best point called **Most Likely Expected Improvement (MLEI)**


#### Paper Number 60
[Annoted PDF in Semantic Scholar](https://www.semanticscholar.org/reader/1460b2073fc915c496021cc40613a87d9b46fa51) 


#### Virtual vs. Real: Trading Off Simulations and Physical Experiments  in Reinforcement Learning with Bayesian Optimization #RL/ModelBased #MutltiFidelity

#### One-liner summary
This paper could be very useful for the multifidelity set up for training that we want to do in RL.
#### Summary
1) Learns cost function J using a GP
2) At each step of training, they choose new $\theta$ (control policy parameters) to try and whether to try it in sim or real using ***entropy*** as measurement
3) To choose the new $\theta$, ***Entropy search*** is used to reduce the uncertainty of the local of minimum $J(\theta)$. Paper provides good explanation of this in **Page 3**
4) $\theta$ is optimized for using Bayesian Optimization
5) The main contribution is that they modify the kernel function of the GP to take multiple information sources - **Page 3 ending**
6) *"Evaluations in simulation (blue  dots) reduce the uncertainty of this blue shaded region, but  reduce only partially the uncertainty about the true cost  (red). In contrast, an evaluation on the real system (red dot)  allows one to learn the true cost J directly, thus reducing  the total uncertainty (red), while some uncertainty about the  variance of Jsim remains (blue). Having uncertainty about  the simulation is by itself irrelevant for the proposed method,  because we solely aim to minimize the performance on the  
physical system."*


#### Paper Number 59
![[2017ChebatorModelBasedAndModelFreeRL.pdf]]
#### Combining Model-Based and Model-Free Updates for Trajectory-Centric Reinforcement Learning #RL/Hybrid

#### One-liner summary:
Combines model based and model free RL but for very simple system using assumptions like linearization - [Link to Semantic Scholar](https://www.semanticscholar.org/reader/a2e2770565665a5d7ba4570934aa1a7882a4e214)


#### Paper Number 58
![[2020KonstantinosRLHandFullTrialsSurvey.pdf]]
#### A Survey on Policy Search Algorithms for Learning Robot Controllers in a Handful of Trials #survey #RL/ModelBased
#### Summary and Quotes
1) This article surveys the literature along these three axes: priors on policy structure and parameters (see Section III), models of expected return (see Section IV), and models of dynamics (see Section V).
2) Gives time taken for training model-free RL techniques
3) model-based PS algorithms scale well with the dimensionality of the policy, but they do not scale with the dimensionality of the state space; and direct PS algorithms scale well with the dimensionality of the state space, but not with the dimensionality of the policy
4) ![[Pasted image 20231229160735.png]]

5) Last section a bit confusing - did not take much from it
#### Paper Number 57
#### Paper
![[2023UtkarshGPUODEandSDEs.pdf]]
#### Title
**Automated Translation and Accelerated Solving of Differential Equations on Multiple GPU Platforms**
#### Authors and affiliation
MIT CSAIl - Julia folks
#### Citations
N.A
#### Keywords
GPU ODE/SDE solvers
#### One-liner summary
ODE and SDE system solver that produces GPU kernels for all kinds of GPUs (Nvidia, AMD, Intel and Apple) and these GPU kernels are also AD compatible.
#### Summary

#### Paper Number 52
![[2023DevosLSBicycleID.pdf]]


#### A least-squares identification method for vehicle cornering stiffness identification from common vehicle sensor data. 
#### Authors and affiliation
T. Devosa,b and F. Naetsa, b
 E2E Lab, Flanders Make@KU Leuven; bDepartment of Mechanical Engineering, KU
Leuven, Belgium
#### Citations
Not published
#### Keywords
Parameter identification/system identification
#### One-liner summary
Identify the linear tire lateral stiffness from a real vehicle dataset using least squares
#### Summary
1) They use available lateral acceleration and yaw rate measurements (converted to yaw acceleration using finite differences) from 2 real vehicles in order to identify the linear tire lateral stiffness for the front and the rear. They remove all the noise in the data using low pass filters
2) The input to the vehicle is the longitudinal velocity which they directly get from the data. In the data this seems to be derived from the wheel speed measurements. The other input is the steering angle which is measure using a steering-mounted encoder.
3) They also don't identify the measurable parameters such as the Inertia's/ mass and directly take these also from the data
4) They have two least square loss functions (one for each sensor) and they weigh them to make one loss function which they optimize. Interestingly enough they find that the yaw rate weight must be about 1000 times more than the lateral velocity weight. 
5) They do not test their calibrated model on unseen maneuvers.


#### Paper Number 53
#### Paper![[2022LeguizamoRLMultiFidelity.pdf]]

#### Title
Deep Reinforcement Learning for Robotic Control with Multi-Fidelity Models
#### Authors and affiliation
David Felipe Leguiza - Iowa
#### Citations
#### Keywords
Multi-fidelity Reinforcement Learning
#### One-liner summary
In this paper, the authors train a DRL policy for a 7 DOF Sawyer robotic arm using a low-fidelity simulation model based on the Denavit-Hartenberg parameters and further fine tune the policy on a high-fidelity Gazebo simulation model. The find that the multi-fidelity transfer learning approach is more efficient on most cases and can be directly transferred onto the real robot without degradation of performance. 
#### Summary


#### Paper Number 54
![[2023DjeumouNeuralSDE.pdf]]
#### Title 
How to Learn and Generalize From Three Minutes of Data: Physics-Constrained and Uncertainty-Aware Neural Stochastic Differential Equations
#### Authors and affiliation
Franck Djeumou - UT Austin
#### Citations 
N.A
#### Keywords
Data driven dynamics
#### One-liner summary
Using Neural Stochastic Differential Equation's to model the dynamics of a hexacopter and then use it with MPC to get amazing performance
#### Summary
1) They don't try to do the impossible and learn the complete dynamics of the hexacopter with a NN. **Instead they provide "physics priors"**

From the paper:

_"The physics-constrained model leverages the structure of 6-dof rigid body dynamics_  
_while parametrizing the aerodynamics forces and moments, the motor command to thrust function,_  
_and (geometric) parameters of the system such as the mass and the inertia matrix"_

Here "parametrizing" means using a NN.

This is what I have been saying we should take as the first step of "learning a model of a vehicle". Learn the maps (which they call "motor comamnd to thrust function") and other forces that are difficult to evaluate on the RHS (In our case these are tire forces or friction forces) rather than learning the entire model. 

2) Since they use stochastic differential equations, the learn the "diffusion term" which **encodes model uncertainty.** This diffusion term is forced to say _"if the current state is close to a state I have seen in training, the uncertainty in predicting the next state is minimum; if the current state is far away from a state seen in training ("far away" is defined using the Eucledian distance between the states in this paper), then increase model uncertainty proportional to the distance"._ 

3) Most importantly, they use their model in a **real world system**. They also recognize that a dynamic model by itself is useless - they thus use it how people in robotics would use it, as a model in MPC or as the environment simulator in Reinforcement learning.  They then show that its performance is great for these applications. 

_**Why do we want to encode model uncertainty in this situation?**_

1) They claim that this helps reinforcement learning algorithms. 

2) Since NN are used, we all know that they are much more accurate near the training data. It thus makes a lot of sense to encode uncertainty in way that it increases far away from the training data.  The way they encode this uncertainty is by using Stochastic Differential Equations - which I think is very neat, simple and extensiable.

#### Paper Number 55
#### Paper
![[2018NagabandiModelBasedAndModelFreeRL.pdf]]
#### Title 
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
#### Authors and affiliation
Anusha Nagabandi - UC Berkley
#### Citations
1015 - Seems like a breakthrough paper
#### Keywords
Model Based RL
#### One-liner summary
They learn an policy using imitation learning from an MPC controller/ NN dynamics set up and use it as a starting point for model free RL.
#### Summary
1) Model based RL seems like the simple concept of learning a parameterized dynamic model and then using that to do control and then reusing that data generated and training the NN again - **A nice flowchart is shown in Page 4**
2) So they do this with an MPC controller and get a good model based controller. They then use all this data to do imitation learning and learn a conditionally Gaussian policy. 
3) They then use this learned policy as the starting point for RL



#### Paper Number 56
#### Paper
![[2020WangRLBenchmark.pdf]]
#### Title 
**Bench-marking Model-Based Reinforcement Learning**
#### Authors and affiliation
Tingwu Wang - UC Berkley
#### Citations
300 odd even though its not published
#### Keywords
Benchmarking model based RL
#### One-liner summary
Benchmarks 11 MBRL and 4 MFRL techniques on 18 OpenGym environments. 
#### Summary
Has a lot of good references for all the different methods out there. All the different methods are well explained in a few lines. Good paper to use as a revision for what was out there towards the end of 2019



