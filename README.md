# Reasoning Models Better Express Their Confidence

### Summary
🙁 LLMs are often overconfident--even when they're wrong. They struggle to express their confidence accurately in their output.

🧐 What about *reasoning models?* Do their slow thinking process help them "know what they know"?

❗️ We find that reasoning models can *dynamically* refine their confidence during CoT (figure below), which leads to superior confidence calibration.

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

## Main Experiment (Section 3)

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
## Analysis (Section 4)
### Linear Regression (Section 4.1)
1. For **reasoning models**:
```
bash evalchemy/reasoning_slope.sh
```

2. For **non-reasoning models**:
```
bash evalchemy/non_reasoning_slope.sh
```
<details>
  <summary>Paths to the segmented CoTs</summary>

</details>


