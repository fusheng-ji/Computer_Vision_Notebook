# Computer_Vision_notebook

![title](/pic/title.jpg)

It's a notebook of Computer Vision which is instructed by Prof. Dr.-Ing. Andreas Geiger in 2021.

Class link: [youtube](https://www.youtube.com/playlist?list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_)

[Course Website](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/) with Slides, Lecture Notes, Problems and Solutions.

[toc]

## Contents

- [L1 Introduction](#l1-introduction)
  - [1.1 Organization](#11-organization)
    - 
  - [1.2 Introduction](#12-introduction)
  - [1.3 History of Computer Vision](#12-history-of-computer-vision)
  
- [L2 Image Formation](#l2-image-formation)
  - [2.1 Primitives and Transformations](#21-primitives-and-transformations)

  - [2.2 Geometric Image Formation](#22-geometric-image-formation)

  - [2.3 Photometric Image Formation](#23-photometric-image-formation)

  - [2.4 Image Sensing Pipeline](#24-image-sensing-pipeline)
  
- [L3 Structure-from-Motion](#l3-structure-from-motion)

  - [3.1 Preliminaries](#31-preliminaries)

  - [3.2 Two-frame Structure-from-Motion](#32-two-frame-structure-from-motion)

  - [3.3 Factorization](#33-factorization)

  - [3.4 Bundle Adjustment](#34-bundle-adjustment)

- [L4 Stereo Reconstruction](#l4-stereo-reconstruction)

  - [4.1 Preliminaries](#41-preliminaries)

  - [4.2 Block Matching](#42-block-matching)

  - [4.3 Siamese Networks](#43-siamese-networks)

  - [4.4 Spatial Regularization](#44-spatial-regularization)

  - [4.5 End-to-End Learning](#45-end-to-end-learning)

- [L5 Probabilistic Graphical Models](#l5-probabilistic-graphical-models)

  - [5.1 Structured Prediction](#51-structured-prediction)

  - [5.2 Markov Random Fields](#52-markov-random-fields)

  - [5.3 Factor Graphs](#53-factor-graphs)

  - [5.4 Belief Propagation](#54-belief-propagation)

  - [5.5 Examples](#55-examples)

- [L6 Applications of Graphical Models](#l6-applications-of-graphical-models)

  - [6.1 Stereo Reconstruction](#61-stereo-reconstruction)

  - [6.2 Multi-View Reconstruction](#62-multi-view-reconstruction)

  - [6.3 Optical Flow](#63-optical-flow)

- [L7 Learning in Graphical Models](#l7-learning-in-graphical-models)

  - [7.1 Conditional Random Fields](#71-conditional-random-fiels)

  - [7.2 Parameter Estimation](#72-parameter-estimation)

  - [7.3 Deep Structured Models](#73-deep-structured-models)

- [L8 Shape-from-X](#l8-shape-from-x)

  - [8.1 Shape-from-Shading](#81-shape-from-shading)

  - [8.2 Photometric Stereo](#82-photometric-stereo)

  - [8.3 Shape-from-X](#83-shape-from-x)

  - [8.4 Volumetric Fusion](#84-volumetric-fusion)

- [L9 Coordinate-based Networks](#l9-coordinate-based-networks)

  - [9.1 Implicit Neural Representations](#91-implicit-neural-representations)

  - [9.2 Differentiable Volumetric Rendering](#92-differentiable-volumetric-rendering)

  - [9.3 Neural Radiance Fields](#93-neural-radiance-fields)

  - [9.4 Generative Radiance Fields](#94-generative-radiance-fields)

- [L10 Recognition](#l10-recognition)

  - [10.1 Image Classification](#101-image-classification)

  - [10.2 Semantic Segmentation](#102-semantic-segmentation)

  - [10.3 Object Detection and Segmentation](#103-object-detection-and-segmentation)

- [L11 Self-Supervised Learning](#l11-self-supervised-learning)

  - [11.1 Preliminaries](#111-preliminaries)

  - [11.2 Task-specific Models](#112-task-specific-models)

  - [11.3 Pretext Tasks](#113-pretext-tasks)

  - [11.4 Contrastive Learning](#114-contrastive-learning)

- [L12 Diverse Topics in Computer Vision](#l12-diverse-topics-in-computer-vision)

  - [12.1 Input Optimization](#121-input-optimization)

  - [12.2 Compositional Models](#122-compositional-models)

  - [12.3 Human Body Models](#123-human-body-models)

  - [12.4 Deepfakes](#124-deepfakes)


## L1 Introduction

### 1.1 Organization

#### Contents

![微信截图_20220214170829](pic/微信截图_20220214170829.png)

#### Course Materials

##### Books

- Szeliski: Computer Vision: Algorithms and Applications
  - https://szeliski.org/Book/
- Hartley and Zisserman: Multiple View Geometry in Computer Vision
  - https://www.robots.ox.ac.uk/~vgg/hzbook/
- Nowozin and Lampert: Structured Learning and Prediction in Computer Vision
  - https://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf
- Goodfellow, Bengio, Courville: Deep Learning
  - http://www.deeplearningbook.org/
- Deisenroth, Faisal, Ong: Mathematics for Machine Learning
  - https://mml-book.github.io
- Petersen, Pedersen: The Matrix Cookbook
  - http://cs.toronto.edu/~bonner/courses/2018s/csc338/matrix_cookbook.pdf

##### Tutorials

- The Python Tutorial
  - https://docs.python.org/3/tutorial/
- NumPy Quickstart
  - https://numpy.org/devdocs/user/quickstart.html
- PyTorch Tutorial
  - https://pytorch.org/tutorials/
- Latex / Overleaf Tutorial
  - https://www.overleaf.com/learn

##### Frameworks / IDEs

- Visual Studio Code
  - https://code.visualstudio.com/
- Google Colab
  - https://colab.research.google.com

##### Courses

- Gkioulekas (CMU): Computer Vision
  - http://www.cs.cmu.edu/~16385/
- Owens (University of Michigan): Foundations of Computer Vision
  - https://web.eecs.umich.edu/~ahowens/eecs504/w20/
- Lazebnik (UIUC): Computer Vision
  - https://slazebni.cs.illinois.edu/spring19/
- Freeman and Isola (MIT): Advances in Computer Vision
  - http://6.869.csail.mit.edu/sp21/
- Seitz (University of Washington): Computer Vision
  - https://courses.cs.washington.edu/courses/cse576/20sp/
- Slide Decks covering Szeliski Book
  - http://szeliski.org/Book/

#### Prerequisites

##### Basic skills

- Basic math skills
  - Linear algebra, probability and information theory. If unsure, have a look at:
    - Goodfellow et al.: [Deep Learning (Book)](https://www.deeplearningbook.org/), Chapters 1-4
    - Luxburg: [Mathematics for Machine Learning (Lecture)](http://www.tml.cs.uni-tuebingen.de/teaching/2020_maths_for_ml/index.php)
    - Deisenroth et al.: [Mathematics for Machine Learning (Book)](https://mml-book.github.io/)
- Basic computer science skills
  - Variables, functions, loops, classes, algorithms
- Basic Python and PyTorch coding skills
  - https://docs.python.org/3/tutorial/
  - https://pytorch.org/tutorials/
- Experience with deep learning. If unsure, have a look at:
  - Geiger: [Deep Learning (Lecture)](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/deep-learning/)

##### Linear Algebra

- vector: $x,y \in \mathbb{R}^n$
- Matrices: $A,B \in \mathbb{R}^{m\times n}$
- Operations: $A^T, A^{-1}, Tr(A), det(A), A+B, AB, Ax, x^Ty$
- Norms: $\left \| x \right \|_1, \left \| x \right \| _2, \left \| x \right \| _\infty, \left \| A \right \| _F$
- SVD: $A = UDV^T$

![微信截图_20220214175244](pic/微信截图_20220214175244.png)

##### Probability and Information Theory

- Probability distributions: $P(X=x)$
- Marginal / conditional: $p(x)=\int p(x,y)dy, p(x,y)=p(x|y)p(y)$
- Bayes rule: $p(x|y)=p(y|x)p(x)/p(y)$
- Conditional independence: $x\perp\!\!\!\perp y|z\Leftrightarrow r(x,y|z)=p(x|z)p(y|z)$
- Expectation: $\mathbb{E}_{x\sim p}[f(x)]=\int_x p(x)f(x)dx$
- Variance: $\mathrm{Var}(f(x))=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]$
- Distributions: Bernoulli, Categorical, Gaussian, Laplace
- Entropy: $H(x)$
- KL Divergence: $D_{KL}(p\parallel q)$

![微信截图_20220214175252](pic/微信截图_20220214175252.png)

##### Deep Learning

- Machine learning basics, linear and logistic regression
- Computation graphs, backpropagation algorithm
- Activation and loss functions, initialization
- Regularization and optimization of deep neural networks
- Convolutional neural networks
- Recurrent neural networks
- Graph neural networks
- Autoencoders and generative adversarial networks

#### Time Management

![微信截图_20220214181539](pic/微信截图_20220214181539.png)

### 1.2 Introduction

### 1.3 History of Computer Vision

## L2 Image Formation

### 2.1 Primitives and Transformations

### 2.2 Geometric Image Formation

### 2.3 Photometric Image Formation

### 2.4 Image Sensing Pipeline

## L3 Structure-from-Motion

### 3.1 Preliminaries

### 3.2 Two-frame Structure-from-Motion

### 3.3 Factorization

### 3.4 Bundle Adjustment

## L4 Stereo Reconstruction

### 4.1 Preliminaries

### 4.2 Block Matching

### 4.3 Siamese Networks

### 4.4 Spatial Regularization

### 4.5 End-to-End Learning

## L5 Probabilistic Graphical Models

### 5.1 Structured Prediction

### 5.2 Markov Random Fields

### 5.3 Factor Graphs

### 5.4 Belief Propagation

### 5.5 Examples

## L6 Applications of Graphical Models

### 6.1 Stereo Reconstruction

### 6.2 Multi-View Reconstruction

### 6.3 Optical Flow

## L7 Learning in Graphical Models

### 7.1 Conditional Random Fields

### 7.2 Parameter Estimation

### 7.3 Deep Structured Models

## L8 Shape-from-X

### 8.1 Shape-from-Shading

### 8.2 Photometric Stereo

### 8.3 Shape-from-X

### 8.4 Volumetric Fusion 

## L9 Coordinate-based Networks

### 9.1 Implicit Neural Representations

### 9.2 Differentiable Volumetric Rendering

### 9.3 Neural Radiance Fields

### 9.4 Generative Radiance Fields

## L10 Recognition

### 10.1 Image Classification

### 10.2 Semantic Segmentation

### 10.3 Object Detection and Segmentation

## L11 Self-Supervised Learning

### 11.1 Preliminaries

### 11.2 Task-specific Models

### 11.3 Pretext Tasks

### 11.4 Contrastive Learning

## L12 Diverse Topics in Computer Vision

### 12.1 Input Optimization

### 12.2 Compositional Models

### 12.3 Human Body Models

### 12.4 Deepfakes
