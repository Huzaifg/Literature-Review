### All old literature review is here
![[highlights.pdf]]


#### Paper Number 105
#### GranularGym: High Performance Simulation for Robotic Tasks with Granular Materials #Year-2023 #GranularSimulation
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/bc574ac79afbee7745f949a040910e230d6e0bf1)

#### One-liner summary
Smashing Granular dynamics simulator with example use of RL
#### Summary
1) #Citable Why fast approx tools are useful
	1) *"Nevertheless, we believe that fast, approximate simulation in complex domains is a powerful tool for the development of robotic autonomy, particularly in closed-loop robotic systems where modeling errors may be accounted for and corrected based on sensor observations."*
2) #Citable Lagrangian particle state representation not good for differentiability
	1) *"The Lagrangian particle state representation is not amenable to useful direct differentiation, due to its permutation invariance. A carefully chosen state representation could pave a path toward meaningful differentiability."*


#### Paper Number 104
#### Learning Terrain-Aware Kinodynamic Model for Autonomous Off-Road Rally Driving With Model Predictive Path Integral Control #Year-2023 #Off-road #MPC
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/37dc6198a18b76b33501ae9819c497a638826cc4)
#### One-liner summary
*"The proposed model generates reliable predictions of 6-degree-of-freedom motion and can even estimate contact interactions without requiring ground truth force data during training. This enables the design of a safe and robust model predictive controller through appropriate cost function design which penalizes sampled trajectories with unstable motion, unsafe interactions, and high levels of uncertainty derived from the model."*
#### Summary
1) #Method Key ideas
	1) **Elevation encoder** that extracts features from a local elevation map $h_t = E_{enc}(M_t)$ where $M_t$ is the local elevation map
	2) Ensemble of **Dynamics Predictive Neural Network** that takes $h_t$ along with linear and angular velocities and angular orientation of the vehicle to predict the change in these. 
	3) This change is then converted to global frame (where it is required) and the vehicle states are integrated with the **Explicit Kinematic Layer**


#### Paper Number 103
#### TerrainNet: Visual Modeling of Complex Terrain for High-speed, Off-road Navigation #Year-2023 #Off-road
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/33fd1bc0a4cd1af507480798e354454e854ce8a2)
#### One-liner summary
*"To this end, we present TerrainNet, a vision-based terrain perception system for semantic and geometric terrain prediction for aggressive, off-road navigation. The approach relies on several key insights and practical considerations for achieving reliable terrain modeling. The network includes a multi-headed output representation to capture fine- and coarse-grained terrain features necessary for estimating traversability. Accurate depth estimation is achieved using self-supervised depth completion with multi-view RGB and stereo inputs. Requirements for real-time performance and fast inference speeds are met using efficient, learned image feature projections. Furthermore, the model is trained on a large scale, real-world off-road dataset collected across a variety of diverse outdoor environments."*

==-> From 4 RGBD images they create a image processing pipeline that generates terrain semantic and elevation maps that is then used in planning==
==-> For planning, they use a kinematic bicycle model and evaluate costs of going through the terrain map based on how much it rolls and pitches (the roll and pitch is just by calculating the height at the wheels lol)==
#### Summary
1) #Method Key facts
	1) TerrainNet takes as input multi-view RGB camera info and Optional Stereo camera inputs
	2) Predicted depth trained using supervised learning. Ground truth depth images from a complete map of the environment using lidar scans
	3) It is 5X faster for reasons I don't understand (something to do with image processing)
2) #Method and #Citable is that terrain traversibility is dependent on mainly 3 factors
	1) Semantics - rocks vs bush
	2) Geometry - Elevation heights
	3) Robot capability - Some robots are more powerful and can climb greater heights
3) #Method They use a multi layer terrain representation that has 
	1) Ground Layer - Stores semantics and elevation statistics at each cell
	2) Ceiling Layer - Models overhanging objects such as canopies and tree branches
4) #Details The images and how they are transformed to give the right maps is shown in page 5 - might be useful for Dan presentation



#### Paper Number 102
#### EVORA: Deep Evidential Traversability Learning for Risk-Aware Off-Road Autonomy #Year-2023 #Adaptive-MPC #Off-road 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/1dffbee25c6ad3713c091497f7e381b16db7f2b4)
#### One-liner summary

*"This work efficiently quantifies both aleatoric and epistemic uncertainties by learning discrete traction distributions and probability densities of the traction predictor’s latent features. Leveraging evidential deep learning, we parameterize Dirichlet distributions with the network outputs and propose a novel uncertainty-aware squared Earth Mover’s distance loss with a closed-form expression that improves learning accuracy and navigation performance"*

==-> For quantifying uncertainty they use "evidential deep learning" to predict distribution of 2 traction parameters which are parameters in their bicycle model==
==-> Terrain semantics which are input to this traction predictor obtained by training offline using a big dataset==

Towards page 4 it became too technical and I did not read further. Might have to come back and understand this in the future

#### Summary
1) #Method Both Aleatoric and Epistemic uncertainties handled with brief in #Details page 2 left side column
2) #Citable The "Related Work" section is really really good here #Details in page 2 Right side column
3) #Method For quantifying uncertainty they use "evidential deep learning"
	1) *"Therefore, we leverage the evidential method proposed in [1] that directly parameterizes the conjugate prior distribution of the target distribution with NN outputs. Specifically, we model the target traction distributions as categorical distributions to capture aleatoric uncertainty, which imply Dirichlet conjugate priors whose concentration parameters, determined by the densities of the traction predictor’s latent features, provide information about epistemic uncertainty"*
4) #Method Main contributions in #Details Page 3 left column
5) #Method There dynamcis is a unicycle model/bicycle model that has as parameters the traction
6) #Method The terrain features they use are
	1) *"Every terrain feature o ∈ O consists of the terrain elevation measured in meter and the one-hot encoding of the semantic label. While we only consider “dirt” and “vegetation” semantic types in simulation, more fine-grained semantic labels are needed in practice"*
7) #Method Terrain semantics obtained by training offline using a big dataset
	1) *"We used PointRend [53] trained on the RUGD off-road navigation dataset [54] with 24 semantic categories to segment RGB images and subsequently projected the semantics onto lidar point clouds."*
8) #Method For Epistemic uncertainty, they measure the density of the latent features using normalzing flows. #Details in Page 5 end and page 6 start

#### Paper Number 101
#### Neural Field Dynamics Model for Granular Object Piles Manipulation #Year-2023 #Data-Driven-Dynamics #Adaptive-MPC 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/7e6df7b43a911b8c766aad7ac5c66947ea6f9e41)

#### One-liner summary
*"To this end, we introduce **Neural Field Dynamics Model (NFD)**, a learning-based dynamics model  
for granular material manipulation. To account for the complex granular dynamics, we leverage the  
insight that the interaction between each granular object is dominated by local friction and contact,  
indicating that each voxel in the scene is only interacting with nearby voxels. To take advantage of  
such sparsity in the transition model and account for translation equivariance, we develop a transition model based on fully convolutional networks (FCN) that uses a unified density-field-based represen-  
tation of both objects and actions. By using differentiable rendering to construct the density-field of  
actions, our model is fully differentiable, allowing it to be integrated with gradient-based trajectory  
optimization methods for planning in complex scenarios, such as pushing piles around obstacles."*


==-> Field based representations are better suited for modelling granular object piles==
==-> Differentiable rasterizer used to render the pusher pose into a one channel image representing the density field of the pushed in the plane==

#### Summary
1) #Citable Particles give good priors on physics but there are drawbacks
	1) *"Alternatively, physics-inspired concepts such as particles lend themselves as strong inductive biases for deep dynamics models. A long line of works approximate a system with a collection of particles and model inter-particle dynamics [3, 4]. However, while conventional particle-based techniques have demonstrated impressive accuracy, their memory and computational costs grow superlinearly with the number of particles [5, 6], posing scalability challenges for their application to granular material manipulations. Moreover, these methods assume that the underlying particles can be tracked, limiting their real-world applicability."*
2) #Method They argue that field based representations are better for modelling granular object piles
	1) *"In this work, we argue that field-based representations are better-suited for modeling granular object piles. By representing the space in which the physical system resides as a density field with discrete sampling positions, we can avoid the challenges associated with modeling interacting particles. Moreover, it facilitates prediction and observation input processing directly in pixel space while providing strong inductive bias, including the sparsity of the dynamics resulting from the locality of the contact mechanics as well as the spatial equivariance of the dynamics."*
3) #Method System state represented by a grid-based density field of granular material. This is how that density field is obtained
	1) *"Specifically, we capture the density field state s by segmenting an RGB image into a one-channel  occupancy grid after an orthographic projection"*
4) #Method Did a nice test to show that FCN models provide strong inductive bias towards granular material dynamics represented as density fields. #Details in page 5
5) #Method Direct sim2real transfer is done here from pyBullet
6) #Citable Good competing methods shown in #Details Page 6 just above section 4.1
7) #Details SoftRasterizer used as the [differentiable renderer](https://www.semanticscholar.org/paper/Soft-Rasterizer%3A-A-Differentiable-Renderer-for-3D-Liu-Li/8b751405526c28245eea5e925a6ede034c287bdb)
8) #Details Shallow version of [UNet ](https://www.semanticscholar.org/paper/U-Net%3A-Convolutional-Networks-for-Biomedical-Image-Ronneberger-Fischer/6364fdaa0a0eccd823a779fcdd489173f938e91a)used as the dynamics model


#### Paper Number 100
#### RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks #Year-2022  #GNN #Data-Driven-Dynamics #Adaptive-MPC 
[PDF on Semantic Scholar here](https://www.semanticscholar.org/reader/35bee2910715971b575f2fd6d4b4874690862bc7)
#### One-liner summary

*"We propose to tackle these challenges by employing a particle-based representation for elasto-plastic objects in a model-based planning framework. Our system, RoboCraft, only assumes access to raw RGBD visual observations. It transforms the sensing data into particles and learns a particle-based dynamics model using graph neural networks (GNNs) to capture the structure of the underlying system."*

==->would it be possible to model the dynamics and manipulate elasto-plastic objects in the real world solely based on RGBD visual observations, without needing particle-to-particle temporal correspondence?==
#### Summary
1) #Citable GNNs seem very good for structure modelling #Details alot of citations on page 1 of this fact
2) #Citable Problem with GNNs is that they require full state information
	1) *"However, most of them require full-state information and a particle-based simulator to provide particle-to-particle correspondence between frames. Such strong supervision is difficult to obtain from raw sensory data, limiting their use in real-world applications. Hence, the natural question to ask here is: would it be possible to model the dynamics and manipulate elasto-plastic objects in the real world solely based on RGBD visual observations, without needing particle-to-particle temporal correspondence?"*
3) #Method This is what they have made and its beautiful
	1) *"Specifically, our framework consists of (1) a perception module that constructs the particle representation of the object by sampling from the reconstructed object mesh, (2) a dynamics model that models the particle interactions using GNNs, and (3) a planning module that uses model-predictive control (MPC) and solves the trajectory optimization problem using gradients from the learned model"*
4) #Method 
	1) First convert RGBD image to point cloud
	2) Convert that point cloud to a mesh
	3) Sample particles within the mesh
	4) Some physics based corrections to this point cloud
	5) Voxel down sampling to get uniform point distribution and to get reasonable points for the GNN
5) #Hardware Cameras used
	1) *"RealSense D415 RGBD cameras are fixed at four locations surrounding the plasticine to capture the RGBD images at 30Hz and 848×480 resolution"*
6) #Method They use random exploration to collect the data
7) #Method In terms of trajectory optimization, the best they found was MPC with GD #Details Page 6
8) #Results Performs better than using an MPM model in the loop - But the GNN here is learning from real world data - On that note it also outperforms itself learnt in sim


#### Paper Number 99
#### Optimistic Active Exploration of Dynamical Systems #Year-2023 #Exploration #RL/ModelBased 
Very theoretical didn't get to the end
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/e66d55565f6cd6336f92be01b4efc7e6a1eb2381)

#### One-liner summary
*"OPAX uses well-calibrated probabilistic models to quantify the epistemic uncertainty about the unknown dynamics. It optimistically—w.r.t. to plausible dynamics—maximizes the information gain between the unknown dynamics and state observations."*

-> To this end, the key question we investigate in this work is: ==how should we interact with the system to learn its dynamics efficiently?==
-> GPs and Bayesian Neural Networks used as Dynamics models
#### Summary
1) #Citable Why exploration is important in MBRL
	1) *"excel in efficiently exploring the dynamical system as they direct the exploration in regions with high rewards. However, due to the directional bias, their underlying learned dynamics model fails to generalize in other areas of the state-action space. While this is sufficient if only one control task is considered, it does not scale to the setting where the system is used to perform several tasks, i.e., under the same dynamics optimized for different reward functions. As a result, when presented with a new reward function, they often need to relearn a policy from scratch, requiring many interactions with the system, or employ multi-task (Zhang and Yang, 2021) or transfer learning (Weiss et al., 2016) methods"*
2) #Method The general method 
	1) *"During each  episode, OPAX plans an exploration policy to gather the most information possible about the system. It learns a statistical dynamics model that can quantify its epistemic uncertainty and utilizes this uncertainty for planning."*
3) #Method Using mutual information to pick policy
	1) *"greedily pick a policy that maximizes the information gain conditioned on the previous observations at each episode"*


#### Paper Number 98
#### Robot Learning with Sensorimotor Pre-training #Year-2023 #Transformer  
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/2b806bc0a075f9088021f7362ffa5b8b86fd75ab)
#### One-liner summary
*"Our model, called RPT, is a Transformer that operates on sequences of sensorimotor tokens. Given a sequence of camera images, proprioceptive robot states, and actions, we encode the sequence into tokens, mask out a subset, and train a model to predict the missing content from the rest. We hypothesize that if a robot can predict the masked-out content it will have acquired a good model of the physical world that can enable it to act."*

==-> Building a foundation model for robotics basically==
#### Summary
1) #Method Images first made into latent features and then tokenized
	1) *"We encode camera images using a pre-trained vision encoder [8] and use latent visua;l representations for sensorimotor sequence learning."*
2) #Method Vision Transformer used to encode image inputs



#### Paper Number 97
#### 	For active exploration, the epistemic uncertainty can be quantified by measuring the ensemble disagreement via Jensen-R ́enyi Divergence. #Year-2023 #RL/ModelBased #Exploration #ProbabilisticDynamicModels 
[PDF from Semantic Scholar](https://www.semanticscholar.org/reader/b9e4e17c2653f561c252a21329327fdbc7c74fea)

#### One-liner summary

*"Our framework uses a probabilistic ensemble neural network for dynamics learning, allowing the quantification of epistemic uncertainty via Jensen-Renyi Divergence. The two opposing tasks of exploration and deployment are optimized through state-of-the-art sampling-based MPC, resulting in efficient collection of training data and successful avoidance of uncertain state-action spaces"*

==-> Jensen-Renyi Divergence used to measure ensemble model diagreement and thus measure the epistemic uncertainty of the model.==

==-> SMPPI (Smooth - MPPI) used to reduce chattering in resulting commands==
#### Summary
1) #Citable Its important to explore while training these NN models
	1) *"Active exploration, in which a robot directs itself to states that yield the highest information gain, is essential for efficient data collection and minimizing human supervision."*
2) #Citable Again great points for Model-free vs Model based RL
	1) *"Firstly, sample efficiency is essential since real-world samples are highly expensive in terms of time, labor, and finances [16]. Secondly, humans are primarily more intuitive about how to incorporate prior knowledge into a model compared to a policy or value function [14, 17]. Lastly, models are task-agnostic and may thus be utilized to optimize arbitrary cost functions, whereas the majority of model-free policies are bounded to a specific task."*
3) #Citable Uncertainty aware deployment and active exploration
	1) *"A spontaneous solution would be to prevent the robot from entering uncertain state-action spaces to evade unpredictable motions"*
	2) *"The exact opposite of the above strategy is to deliberately visit unexplored state-action spaces that provide high uncertainty."*
4) #Method This basically combines uncertainty aware deployment with active exploration
	1) *"In exploration phase, a parallelized ensemble neural network serves as the robot dynamics and outputs the estimated posterior distribution of the next state. In deployment phase, the neural network dynamics trained during the active exploration phase is applied directly to perform uncertainty aware control. Both tasks are optimized using the state-of-the-art sampling-based Model Predictive Contorl (MPC), which, owing to its property, allows the insertion of arbitrary cost functions after training"*
5) #Citable Ensemble of NN's can capture both Aleotric and Epistemic uncertainty
6) #Method VERY GOOD explanation of Ensemble NNs in #Details page 3
7) #Method Jensen-Renyi Divergence used to measure ensemble model diagreement and thus measure the epistemic uncertainty of the model.
	1) *"For active exploration, the epistemic uncertainty can be quantified by measuring the ensemble disagreement via Jensen-R ́enyi Divergence."*

#### Paper Number 96
#### FastRLAP: A System for Learning High-Speed Driving via Deep RL and Autonomous Practicing #Year-2023 #RL #OfflineLearning 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/099c2f6509391246152fbb5c2cd8757dc164ed65)
#### One-liner summary
*"Our system, FastRLAP (faster lap), trains autonomously in the real world, without human interventions, and without requiring any simulation or expert demonstrations. Our system integrates a number of important  components to make this possible: we initialize the representations for the RL policy and value function from a large prior dataset of other robots navigating in other environments (at low speed), which provides a navigation-relevant representation. From here, a sample-efficient online RL method uses a single low-speed user-provided demonstration to determine the desired driving course, extracts a set of navigational checkpoints, and autonomously practices driving through these checkpoints, resetting automatically on collision or failure."*

==-> Use offline data from other robots to pre-train the encoder to extract features of the navigation task==. This speeds up training by a whole lot
==-> A lot of nice hardware related content on page 5==
==-> Shifted tanh activation used for smooth control==
#### Summary
1) #Method Main approach
	1) *"Therefore, our approach combines online RL training with a set of pre-training and initialization steps that aim to maximally transfer prior knowledge to bootstrap real-world RL. Specifically, we aim to use prior data to learn a useful representation of visual observations that captures driving-related features, such as free space and obstacles, while also adapting online to the target domain."*
2) #Method Uses existing data from other robots to bootstrapt and learn quickly online
	1) *" Therefore, our approach combines online RL training with a set of pre-training and initialization steps that aim to maximally transfer prior knowledge to bootstrap real-world RL. Specifically, we aim to use prior data to learn a useful representation of visual observations that captures driving-related features, such as free space and obstacles, while also adapting online to the target domain."*
3) #Method Their state space is interesting
	1) *"Here, V ∈ R128×128×3×3 is a stacked sequence of the last 3 RGB images; v, ω, α ∈ R3 denote the linear velocity, angular velocity, and linear acceleration; the goal g is provided as a relative vector to the next checkpoint, written as a 2D unit vector and a distance; aprev is the previous action."*
4) #Method Ensemble of critics used to regularize
5) #Method State estimation indoors provided by RealSense tracking cameras
	1) *"To estimate state in indoor environments, we use a RealSense T265 tracking camera to provide local visual inertial odometry estimates for the positions of the robot and intermediate checkpoints"*
6) #Method A lot of nice stuff about how the hardware is implemented in #Details on page 5
7) #Details on actuation of motors is provided in page 5
8) #Method **Shifted tanh used to provide smooth actions**
	1) *"To overcome this, we enforce continuity in the policy’s outputs by constraining them to be near the previous action by modifying the action space: (i) instead of the standard tanh activation to limit the action space in [−1, 1] (used by actor), we use a shifted tanh that limits it to the range [aprev − δ, aprev + δ], i.e., near previous action, bounded by δ > 0, and (ii) we append the previous action to the observed state."*


#### Paper Number 95
#### PLANNING GOALS FOR EXPLORATION #Year-2023 #RL/ModelBased #World-Model  
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/5045c2a64a4ea41d8da9ee8eb279498db1ff3d78)

#### One-liner summary

*"We propose “Planning Exploratory Goals” (PEG), a method that sets goals for each training episode to directly optimize an intrinsic exploration reward. PEG first chooses goal commands such that the agent’s goal-conditioned policy, at its current level of training, will end up in states with high exploration potential. It then launches an exploration policy starting at those promising states. To enable this direct optimization, PEG learns world models and adapts sampling-based planning algorithms to “plan goal commands”"*

==-> Provides a nice way for a world model to train better by learning a exploration network alongside a goal reaching network and optimizing for reaching goals that enable exploration==
#### Summary
1) #Method Main contributions
	1) *"We propose a novel paradigm for goal-directed exploration by directly optimizing goal selection to generate trajectories with high exploration value. Next, we show how learned world models permit an effective implementation of goal command planning, by adapting planning algorithms that are often used for low-level action sequence planning"*
2) #Method Very good plot basically explaining the entire method - pick a goal that maximizes the exploration reward from that state - Train $\pi^G$ to reach that goal and $\pi^E$ to explore efficiently from that goal
	1) ![[Pasted image 20240111162736.png]]
	2) *"First command goals that lead the goal policy to states that have high future exploration potential, then explore."*

#### Paper Number 94 
#### Hindsight States: Blending Sim & Real Task Elements for Efficient Reinforcement Learning #Year-2023 #sim2real #RL 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/398ae1bb4de8c043b357c2d095c3bea40bb46525)
#### One-liner summary
*"Here, we leverage the imbalance in complexity of the dynamics to learn more sample-efficiently. We (i) abstract the task into distinct components, (ii) off-load the simple dynamics parts into the simulation, and (iii) multiply these virtual parts to generate more data in hindsight. Our new method, Hindsight States (HiS), uses this data and selects the most useful transitions for training. It can be used with an arbitrary off-policy algorithm."*

==-> All about using sim for simple dynamics to improve sample efficiency==
#### Summary
1) #Citable sim2real gap making people learn on real robots
	1) *"These downsides of sim-to- real approaches accumulate as tasks advance in complexity, thus, necessitating learning many complex tasks partly or completely on the real system."*
2) #Citable Hybrid sim and real (HySR)
	1) *"The idea behind HySR is to keep the complicated parts of the task real, whereas the simpler parts are simulated. This strategy yields significant practical benefits, while facilitating the transfer to the entirely real system."*
3) #Method Complicated combination of sim and real
	1) *"Hindsight States (HiS), is to pair the data of a single real instance with additional data generated by concurrently simulating multiple distinct instances of the virtual part. In the example of robot ball games, our method simulates the effect of the robot’s actions on several virtual balls simultaneously. We relabel the virtual part of each roll-out with this additional virtual data in hindsight."*


#### Paper Number 93
#### Risk-aware Path Planning via Probabilistic Fusion of Traversability Prediction for Planetary Rovers on Heterogeneous Terrains #Year-2023 #Navigation #Off-road 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/7580698f6ac3b3b35eaf9ac13df0e8240198faaf)
#### One-liner summary

*"In this work, we propose a new path planning algorithm that explicitly accounts for such erroneous prediction. The key idea is the probabilistic fusion of distinctive ML models for terrain type classification and slip prediction into a single distribution. This gives us a multi modal slip distribution accounting for heterogeneous terrains and further allows statistical risk assessment to be applied to derive risk-aware traversing costs for path planning."*

#Overview - 
1) Use terrain classifier model
2) Use slip estimation model (GP) based that takes as input terrain class and rover pitch
3) Use CVar to evaluate planned path risk

==-> Conditional value at risk (CVaR) is a nice statistical technique for risk assessment==

==-> The models they use require measurement of terrain pitch angles and wheel slip which are difficult to measure==
#### Summary
1) #Citable They first claim that machine learning is not good enough
	1) *"Machine learning (ML) plays a crucial role in assessing traversability for autonomous rover operations on deformable terrains but suffers from inevitable prediction errors. Especially for heterogeneous terrains where the geological features vary from place to place, erroneous traversability prediction can become more apparent, increasing the risk of unrecoverable rover’s wheel slip and immobilization"*
2) #Citable Because of bad autonomy the travelling of the rover was very slow
	1) *"For instance, the Mars Science Laboratory mission reports an average drive distance of the Curiosity rover being limited to 28.9 meters in one Martian solar day (24 hours 39 min [2]),  even though it can travel up to 15.12 m/h"*
3) #Citable Why navigation on deformable terrain is hard and important very important
	1) *"On extraterrestrial terrain, what turned out to be hazardous for rovers other than apparent obstacles were deformable surfaces. As a known case, the Curiosity rover experienced significant slips on rippled sand at the Hidden Valley, forced to change its route for more solid terrain. Such excessive wheel slips degrade driving speed, increase energy consumption, and eventually cause permanent entrapment in loose, granular materials. Hence, reliable traversability assessment on deformable terrains is essential for autonomous rover operation and, in turn, for faster, more extended rover exploration."*
4) #Method Basically they evaluate risk using the uncertinty in the terrain class predictor and uncertainty in the slip estimate given the terrain class. See Page 1 ending for #Details 
5) ![[Pasted image 20240111151951.png]]
6) #Citable Why probabilistic models are useful
	1) When quantized via probabilistic models, uncertainty provides useful risk assessment tools in ML applications.

7) #Method GP maps terrain geometry to wheel slip
	1) *"To model the unknown relationship between terrain geometry and wheel slip s, we exploit GP, a non-parametric regression approach employing statistical inference to learn dependencies between points in a dataset"*
	2) Dataset uses privileged information *"as rovers measure terrain pitch angles φ and corresponding longitudinal slips s in the past traverse experience."*

8) #Method Pipeline
	1) *"We employ two kinds of pre-trained models: 1) a terrain classifier to pixel-wisely predict terrain classes from appearance imagery and 2) GPs to model classdependent LS functions with geometry information. These models are fused into a single, multi-modal probabilistic slip distribution via mixtures of GPs (MGP)"*
9) 
#### Paper Number 92
#### Neural Optimal Control using Learned System Dynamics #Year-2023 #OptimalControl 
[PDF from Semantic Scholar](https://www.semanticscholar.org/reader/218cf1ccda75676cfdcb5e7008b8d0ea8efee57c)
#### One-liner summary
*"Our approach is to represent the controller and the value function with neural networks, and to train them using loss functions adapted from the Hamilton-Jacobi-Bellman (HJB) equations. In the absence of a known dynamics model, our method first learns the state transitions from data collected by interacting with the system in an offline process. The learned transition function is then integrated to the HJB equations and used to forward simulate the control signals produced by our controller in a feedback loop."*

==-> They use sine activations functions for the dynamics MLP which helps capture the angular shift invariance==
#### Summary
1) #Method Main contributions of the approach
	1) *"To this end, our contributions are two-fold: 1) We propose to approximate the state transitions using neural networks with sinusoidal activation functions and supervise them with numerically computed gradients of the system dynamics, 2) We present a method that integrates the learned dynamics into the HJB equations which are used to train the networks representing the controller and the value function. In our experiments, we show that our method can be used in a variety of systems in a sample efficient fashion."*
2) #Method Using function gradients in supervision
	1) Here they use function gradients also in the loss which is very interesting. #Citable Apparently it leads to learning better state transitions
	2) ![[Pasted image 20240110173528.png]]


#### Paper Number 91
#### ALAN: Autonomously Exploring Robotic Agents in the Real World #Year-2023 #RL/ModelBased #Exploration
[PDF from Semantic Scholar](https://www.semanticscholar.org/reader/f63adcba79ab09c2eed7d22174661be98a018bf4)

#### One-liner summary
*"Thus, we propose ALAN, an autonomously exploring robotic agent, that can perform tasks in the real world with little training and interaction time. This is enabled by measuring environment change, which reflects object movement and ignores changes in the robot position."*

==-> Great ways to enable exploration of agents that we can use while learning a dynamics model==
#### Summary
1) #Method Key intuition behind the method
	1) *"interactions with objects, which cause changes in the visual features of the observations. Thus, seeking to maximize the change in these visual features can be a useful objective for robots to optimize"*
	2) *"Seeking to maximize information related to objects in the environment will lead to much more efficient exploration, since the robot will prioritize actions that lead to richer contact interactions"*
2) #Method Main contributions of the paper
	1) *"The main contribution of this work is ALAN, an efficient real world exploration algorithm, that seeks to take actions that maximize change in the environment, and maximize uncertainty about its internal model of how changes occur in the environment. This approach encourages the robot to interact with objects, and hence collect data relevant to learning manipulation skills faster."*
3) #Method Intrinsic motivation - reward for knowing the flaws within one self
	1) *"When learning a dynamics model of the world, fθ (st+1|st, at), it is possible to use the quality of the model as an intrinsic reward. For instance, Pathak et al. [20] use model prediction error as reward"*
	2) ![[Pasted image 20240110165244.png]]
	3) This requires gradients to optimize since we need $s_{t+1}$ for $r_t$
	4) Another approach as always is to use disagreement between an ensembles of models 
	5) ![[Pasted image 20240110165423.png]]
4) #Method World model described in a better way here which I like more 
	1) ![[Pasted image 20240110165735.png]]

#### Paper Number 90
#### Efficient Preference-Based Reinforcement Learning Using Learned Dynamics Models #Year-2023 #RL/ModelBased #StochasticPlanning 
[PDF from Semantic Scholar](https://www.semanticscholar.org/reader/68c7652c408937ed155bb4289decf7d416b381bc)


#### One-liner summary
*"In particular, we provide evidence that a learned dynamics model offers the following benefits when performing PbRL: (1) preference elicitation and policy optimization require significantly fewer environment interactions than model-free PbRL, (2) diverse preference queries can be synthesized safely and efficiently as a byproduct of standard model-based RL, and (3) reward pre-training based on suboptimal demonstrations can be performed without any environmental interaction. Our paper provides empirical evidence that learned dynamics models enable robots to learn customized policies based on user preferences in ways that are safer and more sample efficient than prior preference learning approaches."*

==-> A nice way to explore is presented. Useful for collecting data to train the dynamics model==
#### Summary
1) #Citable What are preference base RL techniques
	1) *"preference-based reinforcement learning (PbRL) approaches query the user for pairwise preferences over trajectories"*
	2) Basically a reward function is learnt by asking a human to rank between two executed trajectories
2) #Method Nice exploration method for getting data to train dynamics model **random network distillation**
	1) *"However, in many domains, random agent-environment interaction may yield insufficient exploration. To collect diverse data for learning a dynamics model, we leverage random network distillation (RND) [12], a powerful approach for exploration in deep RL that provides reward bonuses based on the error of a neural network predicting the output of a randomly-initialized network."*
#### Paper Number 89
#### Mastering Diverse Domains through World Models #Year-2023 #RL/ModelBased #World-Model  
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/f2d952a183dfb0a1e031b8a3f535d9f8423d7a6e)

#### One-liner summary
*"We present DreamerV3, a general and scalable algorithm based on world models that outperforms previous approaches across a wide range of domains with fixed hyperparameters."*

==-> Should try this with the off-road Chrono problem. Will be interesting to see how it does==
#### Summary
1) #Method Nice description of the neural networks
	1) *"The algorithm consists of 3 neural networks: the world model predicts future outcomes of potential actions, the critic judges the value of each situation, and the actor learns to reach valuable situations."*
2) #Method Key tricks of the trade
	1) *"Specifically, we find that combining KL balancing and free bits enables the world model to learn without tuning, and scaling down large returns without amplifying small returns allows a fixed policy entropy regularizer."*
3) #Method They use something called "symlog" in their loss function which seems to be novel and interesting

#### Paper Number 88
#### Imitation Is Not Enough: Robustifying Imitation with Reinforcement Learning for Challenging Driving Scenarios #Year-2023 #RL/Hybrid #ImitationLearning 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/74af06ac7fa260314064908a8be60d149c55a9ce)

#### One-liner summary
*"In this paper, we show how imitation learning combined with reinforcement learning using simple rewards can substantially improve the safety and reliability of driving policies over those learned from imitation alone. In particular, we train a policy on over 100k miles of urban driving data, and measure its effectiveness in test scenarios grouped by different levels of collision likelihood."*
#### Summary
1) #Citable #Results Advantages of Reinfocement learning over imitation learning
	1) *"Our analysis shows that while imitation can perform well in low-difficulty scenarios that are well-covered by the demonstration data, our proposed approach significantly improves robustness on the most challenging scenarios (over 38% reduction in failures)."*
	2) *"This yields policies that are (1) less vulnerable to covariate shifts and spurious correlations commonly seen in open loop IL [5], [6], and (2) aware of safety considerations encoded in their reward function, but which are only implicit in the demonstrations"*
2) #Citable RL alone is not enough for driving
	1) *"Without accounting for imitation fidelity, driving policies trained with RL may be technically safe but unnatural, and may have a hard time making forward progress in situations that demand human-like driving behavior to coordinate with other agents and follow driving conventions. IL and RL offer complementary strengths: IL increases realism and eases the reward design burden and RL improves safety and robustness, especially in rare and challenging scenarios in the absence of abundant data"*
3) #Citable Some good references for combining IL and RL in #Details page 2
4) #Method Major contributions of the paper
	1) *"We conduct the first large-scale application of a combined IL and RL approach in autonomous driving utilizing large amounts of real-world urban human driving data (over 100k miles) and a simple reward function."*
	2) *"We systematically evaluate its performance and baseline performance by slicing the dataset by difficulty, demonstrating that combining IL and RL improves safety and reliability of policies over those learned from imitation alone (over 38% reduction in safety events on the most difficult bucket)."*



#### Paper Number 87
#### RT-1: ROBOTICS TRANSFORMER FOR REAL-WORLD CONTROL AT SCALE #Year-2023 #TransformerControl

[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/fd1cf28a2b8caf2fe29af5e7fa9191cecfedf84d)
#### One-liner summary
*"We therefore ask: can we train a single, capable, large multi-task backbone model on data consisting of a wide variety of robotic tasks? And does such a model enjoy the benefits observed in other domains, exhibiting zero-shot generalization to new tasks, environments, and objects?"*

-> Transformer used to parameterize the policy $\pi$ and takes as input a sentence describing the task and 6 image history

#### Summary
1) #Method We aim to learn robot policies to solve language-conditioned tasks from vision.
2) #Method $\pi$ which is the transformed is learnt using behavioral cloning


#### Paper Number 86
![[2023PiniAVwithMoE.pdf]]
#### Safe Real-World Autonomous Driving by Learning to Predict and Plan with a Mixture of Experts #AV #Navigation 

#### One-liner summary
*" In this paper, we propose modeling a distribution over multiple future trajectories for both the self-driving vehicle and other road agents, using a unified neural network architecture for prediction and planning. During inference, we select the planning trajectory that minimizes a cost taking into account safety and the predicted probabilities."*

==-> The NN used is a Mixture of Experts Transformer model==
#### Summary
1) #Method Contributions
	1) We propose to model the distribution of future trajectories of agents and the SDV using a mixture of experts in a unified neural network for prediction and planning
	2) We present an efficient and easy to implement decision-making approach that leverages the predicted trajectories and associated probabilities to improve safety by reducing collisions between the SDV and other road agents
	3) We extensively validate our proposal in a realistic closed-loop simulator and deploy it on an SDV driving on public roads, confirming its effectiveness and safety
2) #Citable Covariate shift problem in Imitation learning
	1) *"Although imitation approaches showed significant progress, the covariate shift induced by the policy is still an open problem that can make the model perform poorly during deployment."*

#### Paper Number 85
#### SAM-RL: Sensing-Aware Model-Based Reinforcement Learning via  Differentiable Physics-Based Simulation and Rendering #DifferentiableSimulation #RL/ModelBased  #Year-2023 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/0cc2a916ac1281d3217c775c5bc63844d31a5ba8)
#### One-liner summary
*"In this work, we propose a sensing-aware model-based reinforcement learning system called SAM-RL. Leveraging the differentiable physics-based simulation and rendering, SAM-RL automatically updates the model by comparing rendered images with real raw images and produces the policy efficiently."*

-> The approach seems hard to be possible in real-world because sim updates have to happen in real-time and they seem pretty complicated
#### Summary
1) #Method Three main contributions
	1) *"First, the system no longer requires obtaining a sequence of camera poses at each step, which is extremely time-consuming. Second, compared with using a fixed view, SAM-RL leverages varying camera views with potentially fewer occlusions and offers better estimations of environment states and object status (especially for deformable bodies). The improved quality in object status estimation contributes more effective robotic actions to complete various tasks. Third, ==by comparing rendered and measured (i.e., realworld) images, discrepancies between the simulation and the reality are better revealed and then reduced automatically using gradient-based optimization and differentiable rendering"*==
2) #Method Using differentiable rendering 
	1) Important #Details on how they use the entire pipeline is section 3 2) on page 4
3) #Method The algorithm is very intersing
![[Pasted image 20240108162812.png|300]]

#### Paper Number 84
#### GNM: A General Navigation Model to Drive Any Robot #Navigation #Year-2023 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/3ac400f1ca96a7ccb5a1b7790684abcb00464871)
#### One-liner summary
*"In this paper, we study how a general goal-conditioned model for vision-based navigation can be trained on data obtained from many distinct but structurally similar robots, and enable broad generalization across environments and embodiment's"*

*"We curate 60 hours of navigation trajectories from 6 distinct robots, and deploy the trained GNM on a range of new robots, including an underactuated quadrotor."*
#### Summary
1) #Method The biggest contribution is the dataset and this is impressive
	1) *"The GNM dataset contains over 60 hours of real-world navigation trajectories: a combination of tele-operated and autonomous navigation behaviors collected across 6 distinct robotic platforms, including 4 commercially available platforms (TurtleBot, Clearpath Jackal, Warthog and Spot) and 2 custom platforms (Yamaha Viking ATV, RC Car). The trajectories contain widely varying robot dynamics and top speeds ranging between 0.2 and 10m/s, operating in a diverse set of environments (e.g., office buildings, hallways, suburban, off-road trails, university campus etc.)."*
	 ![[Pasted image 20240108114500.png]]
2) #Citable Why a shared action space is good - It helps generalizing across different robots
	1) *"Learning a common control policy that operates directly on these raw, unstructured outputs can be challenging due to these inconsistencies and high-variance outputs (e.g., speed ∈  [0.2, 10]m/s)."*
	2) *"To this end, we propose using a shared abstraction to allow the goal-reaching policies to operate in a transformed action space that is consistent across robots, making the data points look “similar” and easier to learn common patterns from"*
3) #Method Two key ingredients for generating such a policy is
	1) Shared abstract action space 
	2) Embodiment context that bakes in the capabilities of that particular robot : *"For instance, a TurtleBot can spin in-place but not go over bumps on the road, whereas an RC Car can easily traverse small bumps but has a limited turning radius"*
4) #Method This embodiment context is interesting
	1) *"we use a sequence of consecutive past observations from the robot’s viewpoint to infer a learned embodiment context Ct, and condition the learned policy on this context in addition to the observations. This context contains information about the robot’s configuration and dynamics, which can be used to condition the behavior of the policy."*


#### Paper Number 83
![[2023LeeUncertaintyAwareNavigationOffRoad.pdf]]
#### Learning-based Uncertainty-aware Navigation in 3D Off-Road Terrains #Off-road #Navigation #Year-2023  

#### One-liner summary
*"The proposed algorithm learns the terrain-induced uncertainties from driving
data and encodes the learned uncertainty distribution into the traversability cost for path evaluation. The navigation path is then designed to optimize the uncertainty-aware traversability cost, resulting in a safe and agile vehicle maneuver.****"

==-> A Gaussian process model is used to infer the vehicle-terrain interactions==
Everything again done in sim (Gazebo)
#### Summary
1) #Method The entire approach in simple words
	1) *" First, raw sensor measurements are processed to construct a geometric traversability cost map, and the terrain type map encodes semantic features (e.g., grass, mud, asphalt, etc.). Then, the vehicle-terrain interactions for individual terrain types are respectively learned through Gaussian Process (GP) regression models. By virtue of the GP models, the terrain-induced uncertainties can be expressed by probability distributions. These distributions are used to predict the actual path distributions of the vehicle when the vehicle follows the candidate path, resulting in predictive path distributions [16]. Next, the best path is found by evaluating the cost metrics associated with traversability, rollover risk, and distance to the goal."*
2) #Method Another overview
	1) Mutiple GP models for multiple terrains -> ==The GP model is trained to predict the errors==
	2) Sensors (Camera and Lidar) is used to learn a traversibility cost and terrain type map



#### Paper Number 82
#### RAMP-Net: A Robust Adaptive MPC for Quadrotors via Physics-informed Neural **Network** #Data-Driven-Dynamics #Adaptive-MPC #Year-2023 
[PDF in Semantic Scholar](https://www.semanticscholar.org/reader/d9c67fae3cdc029362bb0f83743bea89866a4909)

#### One-liner summary
*"In this work, we propose a Robust Adaptive MPC framework via PINNs (RAMP-Net), which uses a neural network trained partly from simple ODEs and partly from data. A physics loss is used to learn simple ODEs representing ideal dynamics."*

==-> Everything is completely in a Gazebo sim==
#### Summary
1) #Method PINNs 
	1) *We formulate the ideal system dynamics of a quadrotor  to fit the residual dynamics as a physics loss and use a data loss to capture additional dynamics unaccounted   during mathematical modelling (Section III).  
	2) We train a PINN using the composite loss (sum of the above mentioned loss functions) to approximate the non- linear dynamics of a quadrotor to propose RAMP-Net – a robust adaptive MPC via PINNs (Section IV).  
	3) We perform trajectory tracking of a Hummingbird quadrotor in the Gazebo simulation environment to obtain ∼ 60% lesser tracking error compared to a SOTA regression-based method along with ∼ 11% faster convergence"*
2) #Citable About using parametric uncertain models
	1) *"To reduce the conservatism of robust controllers, adaptive MPC techniques [26], [27] consider   parametric uncertainties over state variables. Such techniques either use functional analysis methods to guarantee closed- loop stability or adapts the controller parameters to mimic a reference model. However, such methods are limited to tackle only parametric uncertainties and tend to overfit to the analytical reference models, a phenomenon known as model drift. Hence, model-based adaptive MPC does not guarantee optimal convergence to true parameters"*
3) #Results #Citable Pure model based MPC suffer when uncertain dynamics added
	1) *"Pure model based robust MPC techniques suffer performance degradation when subjected to uncertain dynamic disturbance"*

#### Paper Number 81
#### Masked World Models for Visual Control #RL/ModelBased #World-Model  #Year-2022
[PDF of Semantic Scholar](https://www.semanticscholar.org/reader/31d629bb161d8199e18b6f2ed7e4ecbda10b6797)
#### One-liner summary
*"In this work, we introduce a visual model-based RL framework that decouples visual representation learning and dynamics learning. Specifically, we train an ==autoencoder with convolutional layers and vision transformers (ViT) to reconstruct pixels given masked convolutional features, and learn a latent dynamics model that operates on the representations from the autoencoder.=="*
#### Summary
1) #Method Key aspects of how the vision and dynamics model are learnt
	1) *"The key idea of MWM is to train an autoencoder that reconstructs visual observations with convolutional feature masking, and a latent dynamics model on top of the autoencoder"*
	2) The masked autoencoder also predicts reward. This helps it learn task relevant features
	3) *"Specifically, we separately update visual representations and dynamics by repeating the iterative processes of (i) training the autoencoder with convolutional feature masking and reward prediction, and (ii) learning the latent dynamics model that predicts visual representations from the autoencoder"*
2) #Method Very nice and short summary on dreamer models and Masked autoencoder in #Details Page 3 and 4

#### Paper Number 80
![[2023SukhijaTrajOptWithLearnedDynamics.pdf]]
#### Gradient-Based Trajectory Optimization With Learned Dynamics #RL/ModelBased #Year-2023  

#### One-liner summary
*"We show that a neural network can model highly nonlinear behaviors
accurately for large time horizons, from data collected in
only 25 minutes of interactions on two distinct robots: (i) the
Boston Dynamics Spot and an (ii) RC car. Furthermore, we
use the gradients of the neural network to perform gradient-
based trajectory optimization. "*

==-> Learn Dynamics model from 25 minutes of on-board data (RNNs and Feed Forward networks)==
#### Summary
1) #Citable GPs scale poorly for large datasets
	1) *"GPs are powerful non-parametric machine learning models that can exhibit strong theoretical guarantees, but they scale poorly for large datasets"*
2) #Citable #Results  They found that an RNN gave better performance than an NN
	1) *" On the left in Fig. 4, we compare the testerror accumulation over open-loop predictions for varying horizons between (i) simple model, (ii) neural network model, and (iii) RNN (GRU) model. The errors of the simple model increase drastically with the horizon length. Nonetheless, the neural network model and the GRU model show better performance, with the GRU giving better results."*
3) #Results There data driven model is bad for long term trajectories -> Fails in open loop settings

#### Paper Number 79
#### VI-IKD: High-Speed Accurate Off-Road Navigation using Learned  Visual-Inertial Inverse Kinodynamics #Navigation #Off-road 
[PDF on Semantic Scholar](https://www.semanticscholar.org/reader/2328d805cd7d01bf1009769a9f684cb78d68070f)

#### One-liner summary
*"we introduce Visual-Inertial Inverse Kinodynamics (VI-IKD), a novel learning  based IKD model that is conditioned on visual information from a terrain patch ahead of the robot in addition to past inertial   information, enabling it to anticipate kinodynamic interactions in the future. We validate the effectiveness of VI-IKD in accurate high-speed off-road navigation experimentally on a scale 1/5 UT-AlphaTruck off-road autonomous vehicle in both indoor and outdoor environments and show that compared to other state-of-the-art approaches, VI-IKD enables more accurate and robust off-road navigation on a variety of different terrains at speeds of up to 3.5m/s."*
#### Summary
1) #Citable Why modelling dynamics and actuation latency's is important at high speeds
	1) *"While ignoring such  effects at low speeds may be acceptable, the combination of actuation latency coupled with kinodynamic responses due to vehicle-terrain interaction can have a magnified effect on the state of a vehicle when travelling at high speeds, and can be catastrophic (e.g., cause collisions) if not accounted for by the controller."*
2) #Method Why they also need a visual sensor in addition to an inertial sensor
	1) *"A model relying on inertial information alone cannot foresee the kinodynamic response at this future position. Unlike an inertial sensor, a visual sensor from an egocentric viewpoint enables perception of the world ahead, providing information about the terrain the vehicle will interact with in the future. We therefore hypothesize that in addition to inertial information from the past, conditioning a learned IKD model on the visual information of the terrain ahead will improve the vehicle’s capability to accurately navigate at high speeds."*
3) #Method Training
![[Pasted image 20240105174402.png|300]]
4) #Method Visual patch extractor and training
	1) One key contribution is the visual patch extractor. See Page 4 for #Details 
	2) Training is done by imitation in the real world with this loss function ($u_t$ is produced by expert)
	![[Pasted image 20240105175520.png]]
5) #Method 
	1) Hausdorf distance used to measure distance between 2 trajectories 
#### Paper Number 78
![[2021IbarzDRLLessonsLearnt.pdf]]
#### How to train your robot with deep reinforcement learning: lessons we have learned #survey #RL 
#### One-liner summary

#### Summary
1) #Citable Why closing Sim2Real gap is so important
	1) *"In simulations, the robots can learn to backflip (Peng et al., 2018a) bicycle stunts (Tan et al., 2014), and even put on clothes (Clegg et al., 2018). In contrast, it is still very challenging to teach robots to perform basic tasks such as walking in the real world. Bridging the reality gap will allow robotics to fully tap into the power of learning. More importantly, bridging the reality gap is important to push the advancement of machine learning for robotics towards the right direction"*
2) #Citable Major cause of sim2real gap is unknown
	1) *"The reality gap is caused by the discrepancy between the simulation and the real-world physics. This error has many sources, including incorrect physical parameters, unmodeled dynamics, and stochastic real environment. However, there is no general consensus about which of these sources plays a more important role. After a large number of experiments with legged robots, both in simulation and on real robots, we found that the actuator dynamics and the lack of latency modeling are the main causes of the model error. ==Developing accurate models for the actuator and latency significantly narrow the reality gap=="*
3) #Citable Model exploitation in MBRL and how its solved
	1) *"model-based RL approaches is model exploitation, i.e., when the model is imperfect in some parts of the state space, and the optimization over actions finds parts of the state space where the model is erroneously optimistic. This can result in poor action selection"*
	2) *"First, we have found that optimization under the model is successful when the data distribution consists of particularly broad distributions over actions and states"*
	3) Another way is model uncertinty
	4) Can make the models adapatable
4) #Citable How to get smooth actions
	1) *"Typically, exploration strategies are realized by adding random noise to the actions. Uncorrelated random noise injected in the action space for exploration can cause jerky motions, which may damage the gearbox and the actuators, and thus is unsafe to execute on the robot. Options for smoothing out jerky actions during exploration include: reward shaping by penalizing jerkiness of the motion, mimicking smooth reference trajectories (Peng et al., 2018a), learning an additive feed- back together with a trajectory generator (Iscen et al., 2018), sampling temporal coherent noise (Haarnoja et al., 2019; Yang et al., 2020), or smoothing the action sequence with low-pass filters. All these techniques work well, although additional manual tuning or user-specified data may be required."*

#### Paper Number 77
![[2021SivaTerrainAdaptation.pdf]]
#### Enhancing Consistent Ground Maneuverability by Robot Adaptation to Complex Off-Road Terrains #Off-road #Navigation
#### One-liner summary
*"Our approach learns offset behaviors in a self-supervised fashion, allowing the robot to compensate
for the inconsistency between the actual and expected behaviors without explicitly modeling the
setbacks, while also adaptively navigating over changing terrain."*

==-> Terrain classification a bunch of papers in page 2==
==-> Terrain adaptation a bunch of papers in page 2==
#### Summary
1) #Citable Learning based methods for terrains
	1) *"learning-based methods can be divided into two broad categories: terrain classification and terrain adaptation. The first category uses a robot’s exteroceptive and proprioceptive sensory data to classify terrain types and estimate traversability for robot navigation over the terrain This category also includes techniques that model terrain complexity for navigation planning [5, 11]. The second category of methods focus on directly generating adaptive behaviors according to terrain in order to successfully complete navigation tasks"*
2) #Method Key Contributions 
	   1) *"The specific novel contributions include:
• We propose a novel mathematical formulation to generate consistent navigational behaviors
by learning offset behaviors in a self-supervised fashion. We also introduce new regulariza-
tion terms to learn important terrain features from multi-sensory observations and fuse them
together to improve robustness of robot adaptation to unstructured terrain.
• We propose a new optimization algorithm to address the formulated regularized optimization problem with dependent variables and non-term regularization terms, which holds a
theoretical guarantee to effectively converge to the global optimal solution."*


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



