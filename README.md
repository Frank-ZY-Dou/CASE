# CASE

Official code for the paper C·ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters}

## Introduction

[//]: # (![teasar]&#40;./assets/fig_teaser.png&#41;)

![fig_teasar.png](assets%2Ffig_teasar.png)
We present C·ASE, an efficient and effective framework that learns conditional Adversarial Skill Embeddings for physics-based characters. Our physically simulated character can learn a diverse repertoire of skills while providing controllability in the form of direct manipulation of the skills to be performed. C·ASE divides the heterogeneous skill motions into distinct subsets containing homogeneous samples for training a low-level conditional model to learn conditional behavior distribution. The skill-conditioned imitation learning naturally offers explicit control over the character's skills after training. The training course incorporates the focal skill sampling, skeletal residual forces, and element-wise feature masking to balance diverse skills of varying complexities, mitigate dynamics mismatch to master agile motions and capture more general behavior characteristics, respectively. Once trained, the conditional model can produce highly diverse and realistic skills, outperforming state-of-the-art models, and can be repurposed in various downstream tasks. In particular, the explicit skill control handle allows a high-level policy or user to direct the character with desired skill specifications, which we demonstrate is advantageous for interactive character animation.

## Data Preparation

C·ASE enables physically simulated characters to master highly varied and extensive skills with high efficiency and effectiveness. Notably, it offers an explicit control handle for directly specifying the desired skill from a diverse and extensive set of skills. Here, a character is instructed by the user to perform a sequence of skills, including kick, jump attack, sword bash, shield bash, and finally, roaring.

## Pre-Training
```angular2html

```

## Interactive Controller Training
```
```

## Interactive Character Animation
```
```


## Citation
