# LIF-RNN-fitting-on-Neuronal-Data
This repository contains Python code that simulates rate-based recurrent neural dynamics and integrates a PyTorch-based model, combining neuroscience-inspired dynamical systems with control theory. The neural dynamics model is capable of simulating time-evolving population activity in response to input cues and optimizing readouts for behaviorally relevant targets (e.g., motor outputs).

# 🧠 Model Description
This repository implements a biologically inspired recurrent neural network (RNN) with the following continuous-time dynamics:

<img width="545" height="250" alt="image" src="https://github.com/user-attachments/assets/ad35a129-eded-4fb8-8020-a475c93fe3a0" />

Recurrent matrix, W, is inferred through imposing constraints on stability by using SoC [1, 3]. You can build W using the code BuildingWeight.py uploaded in this repo.
To infer the optimal initial state of the system, x* and read out of the system, C, that accurately generating the motor outputs, below cost function is designed to be minimized. 

**Minimizing cost-to-go**

<img width="632" height="290" alt="image" src="https://github.com/user-attachments/assets/75d656d3-e1c6-4cbe-bcbc-92c3f7d652c2" />

To ensure system stability, the eigenvalues of the recurrent weight matrix are kept below 1. This implementation is heavily inspired by control theory. For example, the controllability of the network is enhanced by aligning the neural state space with highly controllable directions, identified through eigendecomposition of the controllability Gramian.

# Ongoing Work
## LIF RNN fitting on simulated data
A Python implementation of the cortico-thalamo-cortical loop for optimal movement control, based on T.C. Kao et al., Neuron (2021), will be available here soon.

<img width="428" height="413" alt="image" src="https://github.com/user-attachments/assets/b99ee8c0-683f-4d3a-9d1e-d6f81e937358" />


I synthesized ten directional, arm movement-like trajectories with bell-shaped speed profiles, spanning a range from 0° to 180° [4].
Using these trajectories, I trained the RNN model to infer the optimal neural state, the readout space for movement, and the corresponding thalamic control inputs for two directional movements (0° to 180°).
The model successfully learned to generate with an optimal thalamic inputs to maintain the persistent activity during delay period with minimal energy. This inferred intial state could generate the desired movement for right and left direciton.
<img width="1636" height="1220" alt="simulation_results" src="https://github.com/user-attachments/assets/dcf639af-2121-4dbc-92b0-32c3f5acabfa" />

After training, I re-trained the network to generate novel movements it had never performed before.
This approach tests whether a fully trained RNN can adapt by shaping new initial conditions for unfamiliar movements, or whether a broader reorganization of the network is necessary to learn entirely new motor behaviors.
The model successfully learned additional directional movements and was also able to generate sequential movements within a short time frame by reorganizing the entire network.
<img width="988" height="704" alt="simulation_results_sequential" src="https://github.com/user-attachments/assets/41c011e4-cbf7-4561-9f86-1e01cc9016b9" />

This code will be updated soon. Stay tuned!

# References
[1] Hennequin, G., Vogels, T. P., & Gerstner, W. (2014). Optimal control of transient dynamics in balanced networks supports generation of complex movements. Neuron, 82(6), 1394-1406.

[2] Kao, T. C., Sadabadi, M. S., & Hennequin, G. (2021). Optimal anticipatory control as a theory of motor preparation: A thalamo-cortical circuit model. Neuron, 109(9), 1567-1581.

[3] Jake Stroud's MATLAB code on SoC: https://jakepstroud.github.io/code.html

[4] Kittaka, M., Furui, A., Sakai, H., Morasso, P., & Tsuji, T. (2020). Spatiotemporal Parameterization of Human Reaching Movements Based on Time Base Generator. IEEE Access, 8, 104944-104955.
