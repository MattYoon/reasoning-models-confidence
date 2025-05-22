# Reasoning Models Better Express Their Confidence

[[paper]](https://arxiv.org/abs/2505.14489)

[[tweet (breif overview of the paper)]](https://x.com/dongkeun_yoon/status/1925181877398438068)

### Summary
🙁 LLMs are overconfident even when they are dead wrong. 

🧐 What about reasoning models? Can they actually tell us “My answer is only 60% likely to be correct”?

❗Our paper suggests that they can! Through extensive analysis, we investigate what enables this emergent ability.

<div align="center">
<img src="figure1.png" alt="figure1" width="600"/>
</div>

---
## Installation
```
# clone the repository
pip install -e lm-eval-harness
pip install -e evalchemy
pip install vllm
```

## Section 3: Main Experiment

1. For **reasoning models** that reliably generate "Confidence Reasoning":
```
bash evalchemy/scripts/reasoning_no_force.sh
```

2. For **reasoning models** that do not reliably generate "Confidence Reasoning" (R1 Distill, OR1-Preview, GLM Z1):
```
bash evalchemy/scripts/reasoning_force.sh
```

3. For **non-reasoning models**:
```
bash evalchemy/scripts/non_reasoning.sh
```

4. Finally, use the notebook `results/calculate_metrics.ipynb` to calculate ECE, Brier Score, and AUROC for the outputs.

---
## Section 4: Analysis
### Section 4.1: Linear Regression
1. For **reasoning models**:
```
bash evalchemy/reasoning_slope.sh
```

2. For **non-reasoning models**:
```
bash evalchemy/non_reasoning_slope.sh
```

3. Finally, use the notebook `results/linear_regression.ipynb` to run linear regression on the calibration metrics.

**Change the dataset path and the model name appropriately referring to the list below.**

<details>
  <summary>Paths to the segmented CoTs</summary>

  <b>Reasoning Models</b>  
  - DKYoon/qwen3-think-nonambigqa-slope  
  - DKYoon/qwen3-think-triviaqa-slope  
  - DKYoon/r1-nonambigqa-slope  
  - DKYoon/r1-triviaqa-slope  
  - DKYoon/exaone-deep-nonambigqa-slope  
  - DKYoon/exaone-deep-triviaqa-slope  
  - DKYoon/glm-z1-nonambigqa-slope  
  - DKYoon/glm-z1-triviaqa-slope  


  <b>Non-Reasoning Models</b>  
  - DKYoon/qwen3-non-think-nonambigqa-slope  
  - DKYoon/qwen3-non-think-triviaqa-slope  
  - DKYoon/glm-instruct-nonambigqa-slope  
  - DKYoon/glm-instruct-triviaqa-slope  
  - DKYoon/exaone-instruct-nonambigqa-slope  
  - DKYoon/exaone-instruct-triviaqa-slope  
  - DKYoon/qwen25-nonambigqa-slope  
  - DKYoon/qwen25-triviaqa-slope  
</details>

### Section 4.2: Ablation
```
bash evalchemy/reasoning_ablations.sh
```

The code used to create the ablated CoTs are available in `ablation_data/`.

### Section 4.3: In-context Slow Thinking
```
bash evalchemy/non_reasoning_slow_think.sh
```
The few-shot slow thinking examples are available in `evalchemy/eval/chat_benchmarks/non_reasoning_slow_think/few_shot_prompt.py`

---
## Citation
```
@misc{yoon2025reasoningmodelsbetterexpress,
      title={Reasoning Models Better Express Their Confidence}, 
      author={Dongkeun Yoon and Seungone Kim and Sohee Yang and Sunkyoung Kim and Soyeon Kim and Yongil Kim and Eunbi Choi and Yireun Kim and Minjoon Seo},
      year={2025},
      eprint={2505.14489},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.14489}, 
}
```

