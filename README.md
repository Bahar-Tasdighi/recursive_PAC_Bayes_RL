# Recursive_PAC_Bayes_RL

## Overview

This repository contains code accompanying the paper  
**_Deep Actor-Critics with Tight Risk Certificates_**.

The paper introduces a deep actor–critic framework for risk-constrained reinforcement learning, providing *tight, state-dependent risk certificates* that quantify safety and constraint satisfaction during learning and deployment for Actor critic reinforcement learning agents.

📄 **Paper (arXiv):**  
[//arxiv.org/pdf/2505.19682](https://arxiv.org/pdf/2505.19682?)






## Experiment Logging and Output

All figures and logs are saved automatically in the corresponding `seed_XX/` experiment directory.  

### Summary Figure (`.png`)

A summary figure (`Zplotslogs.png`) is generated for each experiment. It visualizes:

- Training loss history  
- Excess loss (positive/negative)  
- KL divergence  
- PAC-Bayes bound values  
- Posterior vs. prior empirical losses  

### Log File (`.log`)

The log file stores results for all considered baselines in calculatin the PAC-Bayes bounds considered in the paper.   

#### Log Filename Encoding

Each log filename encodes the experimental configuration:

- **NR**: Indicates whether the bound computation is recursive(R) or non-recursive(NR)  
- **informative / noninformative**: Choice of prior  
- **fullvb / finalvb**: Variational Bayesian training applied to all layers or only the final layer of trained model ("lastlayervb", "fullvb")
- **0.02**: Learning rate (configurable for different runs and actor–critic algorithms)  
- **num_portions**: in rhe recursive case, number of portions can be considered eigther 2 or 6.

### Example Usage for Recursuve Baselines
python main.py --model_type lastlayervb --baseline_category recursive  --num_portions 6 

python main.py --model_type fullvb --baseline_category recursive  --num_portions 2 

### Example Usage for Non-Recursuve Baselines
python main.py --model_type fullvb --baseline_category nonrecursive  --loss_type  informative

python main.py --model_type fullvb --baseline_category nonrecursive  --loss_type  noninformative
