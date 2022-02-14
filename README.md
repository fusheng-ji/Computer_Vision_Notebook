# Computer_Vision_notebook

![title](/pic/title.jpg)

It's a notebook of Computer Vision which is instructed by Prof. Dr.-Ing. Andreas Geiger in 2021.

![微信截图_20220214182139](pic/微信截图_20220214182139.png)

Class link: [youtube](https://www.youtube.com/playlist?list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_)

[Course Website](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/) with Slides, Lecture Notes, Problems and Solutions.


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

![微信截图_20220214182311](pic/微信截图_20220214182311.png)

#### Relationships and differences between each conceptions

##### Computer Vision vs. Biological Vision

![微信截图_20220214182413](pic/微信截图_20220214182413.png)

##### Computer Vision vs. Computer Graphics

![微信截图_20220214182428](pic/微信截图_20220214182428.png)

###### Computer Vision is an ill-posed inverse problem

- Many 3D scenes yield the same 2D image
- Additional Constraints (knowledge about world) required

##### Computer Vision vs. Image Processing

![微信截图_20220214190543](pic/微信截图_20220214190543.png)

##### Computer Vision vs. Machine Learning

![微信截图_20220214190631](pic/微信截图_20220214190631.png)

###### The Deep Learning Revolution

![微信截图_20220214190705](pic/微信截图_20220214190705.png)

###### CVPR Submitted and Accepted Papers

![微信截图_20220214190717](pic/微信截图_20220214190717.png)

#### Why is Visual Perception hard?

![微信截图_20220214190908](pic/微信截图_20220214190908.png)

![微信截图_20220214190921](pic/微信截图_20220214190921.png)

##### Challenges: Images are 2D Projections of the 3D World

![微信截图_20220214190942](pic/微信截图_20220214190942.png)

![微信截图_20220214190953](pic/微信截图_20220214190953.png)

![微信截图_20220214191007](pic/微信截图_20220214191007.png)

##### Challenges: Viewpoint Variation

![微信截图_20220214191251](pic/微信截图_20220214191251.png)

##### Challenges: Deformation

![微信截图_20220214191259](pic/微信截图_20220214191259.png)

##### Challenges: Occlusion

![微信截图_20220214191309](pic/微信截图_20220214191309.png)

##### Challenges: illumination

![微信截图_20220214191322](pic/微信截图_20220214191322.png)

![微信截图_20220214191332](pic/微信截图_20220214191332.png)

![微信截图_20220214191344](pic/微信截图_20220214191344.png)

##### Challenges: Motion

![微信截图_20220214191352](pic/微信截图_20220214191352.png)

##### Challenges: Perception vs. Measurement

![微信截图_20220214191402](pic/微信截图_20220214191402.png)

![微信截图_20220214191410](pic/微信截图_20220214191410.png)

![微信截图_20220214191418](pic/微信截图_20220214191418.png)

##### Challenges: Local Ambiguities

![微信截图_20220214191427](pic/微信截图_20220214191427.png)

![微信截图_20220214191815](pic/微信截图_20220214191815.png)

![微信截图_20220214191823](pic/微信截图_20220214191823.png)

##### Challenges: Intra Class Variation

![微信截图_20220214191831](pic/微信截图_20220214191831.png)

##### Challenges: Number of Object Categories

![微信截图_20220214191839](pic/微信截图_20220214191839.png)

### 1.3 History of Computer Vision

#### Credits

- Svetlana Lazebnik (UIUC): Computer Vision: Looking Back to Look Forward
  - https://slazebni.cs.illinois.edu/spring20/
- Steven Seitz (Univ. of Washington): 3D Computer Vision: Past, Present, and Future
  - http://www.youtube.com/watch?v=kyIzMr917Rc
  - http://www.cs.washington.edu/homes/seitz/talks/3Dhistory.pdf

#### Overview

![微信截图_20220214192929](pic/微信截图_20220214192929.png)

##### Waves of development

- 1960-1970: Blocks Worlds, Edges and Model Fitting
- 1970-1981: Low-level vision: stereo, flow, shape-from-shading
- 1985-1988: Neural networks, backpropagation, self-driving
- 1990-2000: Dense stereo and multi-view stereo, MRFs
- 2000-2010: Features, descriptors, large-scale structure-from-motion
- 2010-now: Deep learning, large datasets, quick growth, commercialization

#### A Brief History of Computer Vision

##### 1957: Stereo

![微信截图_20220214192941](pic/微信截图_20220214192941.png)

##### 1958-1962: Rosenblatt's Perceptron

![微信截图_20220214192950](pic/微信截图_20220214192950.png)

![微信截图_20220214193004](pic/微信截图_20220214193004.png)

##### 1963: Larry Robert's Blocks World

![微信截图_20220214193016](pic/微信截图_20220214193016.png)

##### 1966: MIT Summer Vision Project

![微信截图_20220214193030](pic/微信截图_20220214193030.png)

##### 1969: Minsky and Papert publish book

![微信截图_20220214193039](pic/微信截图_20220214193039.png)

##### 1970: MIT Copy Demo

![微信截图_20220214193049](pic/微信截图_20220214193049.png)

##### 1970: Shape from Shading

![微信截图_20220214193057](pic/微信截图_20220214193057.png)

##### 1978: Intrinsic Images

![微信截图_20220214193106](pic/微信截图_20220214193106.png)

##### 1980: Photometric Stereo![微信截图_20220214193116](pic/微信截图_20220214193116.png)

##### 1981: Essential Matrix

![微信截图_20220214193124](pic/微信截图_20220214193124.png)

##### 1981: Binocular Scanline Stereo

![微信截图_20220214193132](pic/微信截图_20220214193132.png)

##### 1981: Dense Optical Flow

![微信截图_20220214193142](pic/微信截图_20220214193142.png)

##### 1984: Markov Random Fields

![微信截图_20220214193150](pic/微信截图_20220214193150.png)

##### 1980s: Part-based Models

![微信截图_20220214193158](pic/微信截图_20220214193158.png)

##### 1986: Backpropagation Algorithm

![微信截图_20220214193208](pic/微信截图_20220214193208.png)

##### 1986: Self-Driving Car VaMoRs

![微信截图_20220214193215](pic/微信截图_20220214193215.png)

##### 1988: Self-Driving Car ALVIMN

![微信截图_20220214194355](pic/微信截图_20220214194355.png)

##### 1992: Structure-from-Motion

![微信截图_20220214194406](pic/微信截图_20220214194406.png)

##### 1992: Iterative Closest Points

![微信截图_20220214194415](pic/微信截图_20220214194415.png)

##### 1996: Volumetric Fusion

![微信截图_20220214194424](pic/微信截图_20220214194424.png)

##### 1998: Multi-View Stereo

![微信截图_20220214194432](pic/微信截图_20220214194432.png)

##### 1998: Stereo with Graph Cuts

![微信截图_20220214194442](pic/微信截图_20220214194442.png)

##### 1998: Convolutional Neural Networks

![微信截图_20220214194451](pic/微信截图_20220214194451.png)

##### 1999: Morphable Models

![微信截图_20220214194504](pic/微信截图_20220214194504.png)

##### 1999: SIFT

![微信截图_20220214194511](pic/微信截图_20220214194511.png)

##### 2006: Photo Tourism

![微信截图_20220214194520](pic/微信截图_20220214194520.png)

##### 2007: PMBS

![微信截图_20220214194531](pic/微信截图_20220214194531.png)

##### 2009: Building Rome in a Day

![微信截图_20220214194541](pic/微信截图_20220214194541.png)

##### 2011: Kinect

![微信截图_20220214194549](pic/微信截图_20220214194549.png)

##### 2009-2012: ImageNet and AlexNet

![微信截图_20220214194557](pic/微信截图_20220214194557.png)

##### 2002-now: Golden Age of Datasets

![微信截图_20220214194612](pic/微信截图_20220214194612.png)

##### 2017-now: Syntheic Data

![微信截图_20220214194629](pic/微信截图_20220214194629.png)

![微信截图_20220214194820](pic/微信截图_20220214194820.png)

##### 2014: Visualization

![微信截图_20220214194646](pic/微信截图_20220214194646.png)

##### 2014: Adversarial Examples

![微信截图_20220215004115](pic/微信截图_20220215004115.png)

##### 2014: Generative Adversarial Networks

![微信截图_20220215004123](pic/微信截图_20220215004123.png)

##### 2014: DeepFace

![微信截图_20220215004133](pic/微信截图_20220215004133.png)

##### 2014: 3D Scene Understanding

![微信截图_20220215004144](pic/微信截图_20220215004144.png)

##### 2014: 3D Scanning

![微信截图_20220215004152](pic/微信截图_20220215004152.png)

##### 2015: Deep Reinforcement Learning

![微信截图_20220215004203](pic/微信截图_20220215004203.png)

##### 2016: Style Transfer

![微信截图_20220215004219](pic/微信截图_20220215004219.png)

##### 2015-2017: Semantic Segmentation

![微信截图_20220215004228](pic/微信截图_20220215004228.png)

##### 2017: Mask R-CNN

![微信截图_20220215004238](pic/微信截图_20220215004238.png)

##### 2017: Image Captioning

![微信截图_20220215004247](pic/微信截图_20220215004247.png)

##### 2018: Human Shape and Pose

![微信截图_20220215004255](pic/微信截图_20220215004255.png)

##### 2016-2020: 3D Deep Learning

![微信截图_20220215004307](pic/微信截图_20220215004307.png)

##### Applications and Commercial Products

![微信截图_20220215004319](pic/微信截图_20220215004319.png)

##### Current Challenges

- Un-/Self-Supervised Learning
- Interactive learning
- Accuracy (e.g., self-driving)
- Robustness and generalization
- Inductive biases
- Understanding and mathematics
- Memory and compute
- Ethics and legal questions

## L2 Image Formation

### 2.1 Primitives and Transformations

- Geometric primitives are the basic building blocks used to describe 3D shapes
- In this unit, we introduce points, lines and planes
- Furthermore, the most basic transformations are discussed
- This unit covers the topics of the Szeliski book, chapter 2.1
- A more exhaustive introduction can be found in the book:
  Hartley and Zisserman: Multiple View Geometry in Computer Vision

#### 2D Points

2D points can be written in inhomogeneous coordinates as $x=\begin{pmatrix}x\\y\end{pmatrix}\in \mathbb{R}^2$, or in homogeneous coordinates as $\tilde{x}=\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}\in \mathbb{P}^2$, where $\mathbb{P}^2=\mathbb{R}^3\setminus \left \{  0,0,0\right \}$ is called projective space.

**Remark**: Homogeneous vectors that differ only by scale are considered equivalent and define an equivalence class. 👉Homogeneous vectors are defined only up to scale.

An inhomogeneous vector x is converted to a homogeneous vector x as follows $\tilde{x}=\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}=\begin{pmatrix}x\\y\\1\end{pmatrix}=\begin{pmatrix}x\\1\end{pmatrix}=\bar{x}$

with augmented vector $\bar{x}$.To convert in the opposite direction we divide by $\tilde{w}$ : $\bar{x}=\begin{pmatrix}x\\1\end{pmatrix}=\begin{pmatrix}x\\y\\1\end{pmatrix}=\frac{1}{\tilde{w}}\tilde{x}=\frac{1}{\tilde{w}}\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}=\begin{pmatrix}\frac{\tilde{x}}{\tilde{w}}\\\frac{\tilde{y}}{\tilde{w}}\\1\end{pmatrix}$

Homogeneous points whose last element is $\tilde{w}=0$ are called ideal points or points at infinity. These points can’t be represented with inhomogeneous coordinates!

![Snipaste_2022-02-15_01-13-19](pic/Snipaste_2022-02-15_01-13-19.png)

![Snipaste_2022-02-15_01-13-39](pic/Snipaste_2022-02-15_01-13-39.png)

![Snipaste_2022-02-15_01-13-47](pic/Snipaste_2022-02-15_01-13-47.png)

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
