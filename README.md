# Defection-Free Collaboration between Competitors in a Learning System

This repository contains the code for the paper [Defection-Free Collaboration between Competitors in a Learning System](https://arxiv.org/abs/2406.15898) by Mariel Werner, Sai Praneeth Karimireddy, and Michael I. Jordan.

## Abstract

We study collaborative learning systems in which the participants are competitors who will defect from the system if they lose revenue by collaborating. As such, we frame the system as a duopoly of competitive firms who are each engaged in training machine-learning models and selling their predictions to a market of consumers. We first examine a fully collaborative scheme in which both firms share their models with each other and show that this leads to a market collapse with the revenues of both firms going to zero. We next show that one-sided collaboration in which only the firm with the lower-quality model shares improves the revenue of both firms. Finally, we propose a more equitable, *defection-free* scheme in which both firms share with each other while losing no revenue, and we show that our algorithm converges to the Nash bargaining solution.

## Code Execution

Simply run the files to reproduce the plots in the paper.
```
python3 graphs_dfcl.py
python3 mnist_dfcl.py
```
