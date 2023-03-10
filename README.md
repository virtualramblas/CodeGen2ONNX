# CodeGen2ONNX
Porting the CodeGen Language Model to ONNX format.  
### What is CodeGen?
[CodeGen](https://github.com/salesforce/CodeGen) is an Open Source model for program synthesis, competitive with OpenAI Codex. Pre-trained models of various sizes (350M, 2B, 6B and 16B parameters) are available for it. The pre-trained models labelled as *multi* have been trained to generate code in multiple programming languages (C, C++, Go, Java, JavaScript, Python), while those labelled as *mono* are trained to generate code only in Python. While most of the work present here should apply to both families, this repo focuses on the *mono* models only.  
### What is ONNX?
[ONNX](https://onnx.ai/) is an open format built to represent machine learning models, which aims to make interoperability across diverse ML/DL frameworks and performance maximization across diverse hardware accelerators easier.  
### Why this repo?
This repo was born following a need to generate Python code starting from natural language using an Open Source model such as CodeGen in environments having computational power constraints.   
  
*** Code Coming soon! ***   
