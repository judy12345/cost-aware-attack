# cost-aware-attack
Implementation for **paper 'Cost Aware Untargeted Poisoning Attack against Graph Neural Networks'**

In this work, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins.

The code is based on the **deeprobust** (https://deeprobust.readthedocs.io/en/latest/graph/pyg.html)
To reproduce the attack performance:
    Run ```Test_attack.py``` to generate the attacked graph and test the attack perforamnce.

Please find our paper at cost_aware_full.pdf

