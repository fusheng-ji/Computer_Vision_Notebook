---
typora-root-url: ./
---

# Computer_Vision_notebook

![title](pic/title.jpg)

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
- In this unit, we introduce **points, lines and planes**
- Furthermore, the most basic transformations are discussed
- This unit covers the topics of [the Szeliski book](https://szeliski.org/Book/), chapter 2.1
- A more exhaustive introduction can be found in the book:
  Hartley and Zisserman: Multiple View Geometry in Computer Vision

#### 2D Points

2D points can be written in inhomogeneous coordinates as $\mathrm{x}=\begin{pmatrix}x\\y\end{pmatrix}\in \mathbb{R}^2$, or in homogeneous coordinates as $\tilde{\mathrm{x}}=\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}\in \mathbb{P}^2$, where $\mathbb{P}^2=\mathbb{R}^3\setminus \left \{  0,0,0\right \}$ is called projective space.

A tilde symbol denotes a homogeneous coordinate.

**Remark**: Homogeneous vectors that differ only by scale are considered equivalent and define an equivalence class. 👉Homogeneous vectors are defined only up to scale.

An inhomogeneous vector x is converted to a homogeneous vector x as follows $\tilde{\mathrm{x}}=\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}=\begin{pmatrix}x\\y\\1\end{pmatrix}=\begin{pmatrix}\mathrm{x}\\1\end{pmatrix}=\bar{\mathrm{x}}$

with augmented vector $\bar{\mathrm{x}}$. To convert in the opposite direction we divide by $\tilde{w}$ : $\bar{\mathrm{x}}=\begin{pmatrix}\mathrm{x}\\1\end{pmatrix}=\begin{pmatrix}x\\y\\1\end{pmatrix}=\frac{1}{\tilde{w}}\tilde{\mathrm{x}}=\frac{1}{\tilde{w}}\begin{pmatrix}\tilde{x}\\\tilde{y}\\\tilde{w}\end{pmatrix}=\begin{pmatrix}\frac{\tilde{x}}{\tilde{w}}\\\frac{\tilde{y}}{\tilde{w}}\\1\end{pmatrix}$

**Remark**: Homogeneous points whose last element is $\tilde{w}=0$ are called **ideal points** or **points at infinity**. These points can’t be represented with inhomogeneous coordinates!

![Snipaste_2022-02-15_01-13-19](pic/Snipaste_2022-02-15_01-13-19.png)

![Snipaste_2022-02-15_01-13-39](pic/Snipaste_2022-02-15_01-13-39.png)

![Snipaste_2022-02-15_01-13-47](pic/Snipaste_2022-02-15_01-13-47.png)

#### 2D Line

2D lines can also be expressed using homogeneous coordinates $\tilde{\mathrm{I}}$

![2022-02-19_10-33](pic/2022-02-19_10-33.png)

#### Cross Product

So the homogeneous coordinates allow us to represent infinity point and the relationships between points and lines.

![2022-03-13_23-27](/pic/2022-03-13_23-27.png)

#### 2D Line Arithmetic

![2022-03-13_23-28](/pic/2022-03-13_23-28.png)

![2022-03-13_23-28_1](/pic/2022-03-13_23-28_1.png)

![2022-03-13_23-29](/pic/2022-03-13_23-29.png)

#### 2D Conics

![2022-03-14_11-54](/pic/2022-03-14_11-54.png)

#### 3D Points

![2022-03-14_11-55](/pic/2022-03-14_11-55.png)

#### 3D Planes

![2022-03-14_11-55_1](/pic/2022-03-14_11-55_1.png)

![2022-03-14_11-55_2](/pic/2022-03-14_11-55_2.png)

#### 3D Lines

![2022-03-14_12-09](/pic/2022-03-14_12-09.png)

#### 3D Quadircs

![2022-03-14_12-09_1](/pic/2022-03-14_12-09_1.png)

![2022-03-14_12-09_2](/pic/2022-03-14_12-09_2.png)

#### 2D Transformations

![2022-03-14_12-10](/pic/2022-03-14_12-10.png)

![2022-03-14_12-10_1](/pic/2022-03-14_12-10_1.png)

![2022-03-14_12-10_2](/pic/2022-03-14_12-10_2.png)

![2022-03-14_12-11](/pic/2022-03-14_12-11.png)

![2022-03-14_12-11_1](/pic/2022-03-14_12-11_1.png)

#### 2D Transformations on Co-Vectors

![2022-03-14_12-12](/pic/2022-03-14_12-12.png)

#### Overview of 2D Transformations

![2022-03-14_12-20](/pic/2022-03-14_12-20.png)

#### Overview of 3D Transformations

![2022-03-14_12-20_1](/pic/2022-03-14_12-20_1.png)

#### Direct Linear Transform for Homography Estimation

![2022-03-14_12-21](/pic/2022-03-14_12-21.png)

![2022-03-14_12-21_1](/pic/2022-03-14_12-21_1.png)

##### Application: Panorama Stitching

![2022-03-14_12-24](/pic/2022-03-14_12-24.png)

### 2.2 Geometric Image Formation

#### Origins of the Pinhole Camera

![2022-03-14_12-26](/pic/2022-03-14_12-26.png)

![2022-03-14_12-26_1](/pic/2022-03-14_12-26_1.png)

![2022-03-14_12-28](/pic/2022-03-14_12-28.png)

https://www.abelardomorell.net/camera-obscura

![2022-03-14_12-29](/pic/2022-03-14_12-29.png)

#### Projection Models

![2022-03-14_12-29_1](/pic/2022-03-14_12-29_1.png)

![2022-03-14_12-29_2](/pic/2022-03-14_12-29_2.png)

##### Orthographic Projection

![2022-03-14_12-32](/pic/2022-03-14_12-32.png)

![2022-03-14_12-32_1](/pic/2022-03-14_12-32_1.png)

##### Scaled Orthographic Projection

![2022-03-14_12-33](/pic/2022-03-14_12-33.png)

##### Perspective Projection

![2022-03-14_12-48](/pic/2022-03-14_12-48.png)

![2022-03-14_12-48_1](/pic/2022-03-14_12-48_1.png)

![2022-03-14_12-48_2](/pic/2022-03-14_12-48_2.png)

![2022-03-14_12-49](/pic/2022-03-14_12-49.png)

#### Chaining Transformations

![2022-03-14_12-54](/pic/2022-03-14_12-54.png)

#### Full Rank Representation

![2022-03-14_12-55](/pic/2022-03-14_12-55.png)

#### Lens Distortion

![2022-03-14_12-55_1](/pic/2022-03-14_12-55_1.png)

![2022-03-14_12-55_2](/pic/2022-03-14_12-55_2.png)

### 2.3 Photometric Image Formation

#### Photometric Image Formation

![2022-03-14_12-58](/pic/2022-03-14_12-58.png)

#### Rendering Equation

![2022-03-14_12-58_1](/pic/2022-03-14_12-58_1.png)

#### Diffuse and Specular Reflection

![2022-03-14_12-59](/pic/2022-03-14_12-59.png)

![2022-03-14_12-59_1](/pic/2022-03-14_12-59_1.png)

#### BRDF Examples

![2022-03-14_13-00](/pic/2022-03-14_13-00.png)

#### Fresnel Effect

![2022-03-14_13-00_1](/pic/2022-03-14_13-00_1.png)

#### Global Illumination

![2022-03-14_13-01](/pic/2022-03-14_13-01.png)

#### Why Camera Lenses?

![2022-03-14_13-03](/pic/2022-03-14_13-03.png)

http://www.pauldebevec.com/Pinhole/

![2022-03-14_13-03_1](/pic/2022-03-14_13-03_1.png)

#### Optics

![2022-03-14_13-04](/pic/2022-03-14_13-04.png)

#### Thin Lens Model

![2022-03-14_13-04_1](/pic/2022-03-14_13-04_1.png)

#### Depth of Field (DOF)

![2022-03-14_13-05](/pic/2022-03-14_13-05.png)

![2022-03-14_13-05_1](/pic/2022-03-14_13-05_1.png)

![2022-03-14_13-05_2](/pic/2022-03-14_13-05_2.png)

![2022-03-14_13-05_3](/pic/2022-03-14_13-05_3.png)

#### Chromatic Aberration

![2022-03-14_13-06](/pic/2022-03-14_13-06.png)

![2022-03-14_13-06_1](/pic/2022-03-14_13-06_1.png)

#### Vignetting

![2022-03-14_13-06_2](/pic/2022-03-14_13-06_2.png)

![2022-03-14_13-07](/pic/2022-03-14_13-07.png)

### 2.4 Image Sensing Pipeline

#### Image Sensing Pipeline

![2022-03-14_13-09](/pic/2022-03-14_13-09.png)

#### Shutter

![2022-03-14_13-09_1](/pic/2022-03-14_13-09_1.png)

#### Sensor

![2022-03-14_13-10](/pic/2022-03-14_13-10.png)

#### Color Filter Arrays

![2022-03-14_13-10_1](/pic/2022-03-14_13-10_1.png)

![2022-03-14_13-11](/pic/2022-03-14_13-11.png)

![2022-03-14_13-11_1](/pic/2022-03-14_13-11_1.png)

#### Color Spaces

![2022-03-14_13-12](/pic/2022-03-14_13-12.png)

#### Gamma Compression

![2022-03-14_13-12_1](/pic/2022-03-14_13-12_1.png)

#### Image Compression

![2022-03-14_13-12_2](/pic/2022-03-14_13-12_2.png)

## L3 Structure-from-Motion

### 3.1 Preliminaries

#### Camera Calibration

![2022-03-14_13-25](/pic/2022-03-14_13-25.png)

![2022-03-14_13-25_1](/pic/2022-03-14_13-25_1.png)

![2022-03-14_13-26](/pic/2022-03-14_13-26.png)

![2022-03-14_13-26_1](/pic/2022-03-14_13-26_1.png)

![2022-03-14_13-26_2](/pic/2022-03-14_13-26_2.png)

#### Feature Detection and Description

##### Point Features

![2022-03-14_13-27](/pic/2022-03-14_13-27.png)

![2022-03-14_13-27_1](/pic/2022-03-14_13-27_1.png)

##### Scale Invariant Feature Transform (SIFT)

![2022-03-14_13-36](/pic/2022-03-14_13-36.png)

![2022-03-14_13-37](/pic/2022-03-14_13-37.png)

![2022-03-14_13-37_1](/pic/2022-03-14_13-37_1.png)

![2022-03-14_13-38](/pic/2022-03-14_13-38.png)

![2022-03-14_13-38_1](/pic/2022-03-14_13-38_1.png)

![2022-03-14_13-39](/pic/2022-03-14_13-39.png)

### 3.2 Two-frame Structure-from-Motion

#### Epipolar Geometry

![2022-03-14_13-44](/pic/2022-03-14_13-44.png)

![2022-03-14_13-44_1](/pic/2022-03-14_13-44_1.png)

![2022-03-14_13-45](/pic/2022-03-14_13-45.png)

![2022-03-14_13-45_1](/pic/2022-03-14_13-45_1.png)

![2022-03-14_13-45_2](/pic/2022-03-14_13-45_2.png)

##### Estimating the Epipolar Geometry

![2022-03-14_13-47](/pic/2022-03-14_13-47.png)

![2022-03-14_13-47_1](/pic/2022-03-14_13-47_1.png)

##### Estimating the Epipolar Geometry with unknown Intrinsics

![2022-03-14_13-49](/pic/2022-03-14_13-49.png)

![2022-03-14_13-50](/pic/2022-03-14_13-50.png)

#### Trianulation

![2022-03-14_13-55](/pic/2022-03-14_13-55.png)

![2022-03-14_13-55_1](/pic/2022-03-14_13-55_1.png)

![2022-03-14_13-55_2](/pic/2022-03-14_13-55_2.png)

##### Triangulation Uncertainty

![2022-03-14_13-56](/pic/2022-03-14_13-56.png)

### 3.3 Factorization

#### Orthographic Factorization

![2022-03-14_13-57](/pic/2022-03-14_13-57.png)

![2022-03-14_13-57_1](/pic/2022-03-14_13-57_1.png)

![2022-03-14_13-57_2](/pic/2022-03-14_13-57_2.png)

![2022-03-14_13-57_3](/pic/2022-03-14_13-57_3.png)

![2022-03-14_13-57_4](/pic/2022-03-14_13-57_4.png)

![2022-03-14_13-57_5](/pic/2022-03-14_13-57_5.png)

![2022-03-14_13-58](/pic/2022-03-14_13-58.png)

![2022-03-14_13-58_1](/pic/2022-03-14_13-58_1.png)

![2022-03-14_13-58_2](/pic/2022-03-14_13-58_2.png)

![2022-03-14_13-58_3](/pic/2022-03-14_13-58_3.png)

Tomasi and Kanade: Shape and motion from image streams under orthography: a factorization method. IJCV, 1992.

![2022-03-14_13-58_4](/pic/2022-03-14_13-58_4.png)

##### Perspective Factorization

![2022-03-14_14-00](/pic/2022-03-14_14-00.png)

### 3.4 Bundle Adjustment

#### Bundle Adjustment

![2022-03-14_14-01](/pic/2022-03-14_14-01.png)

![2022-03-14_14-01_1](/pic/2022-03-14_14-01_1.png)

![2022-03-14_14-01_2](/pic/2022-03-14_14-01_2.png)

##### Challenges of Bundle Adjustment

![2022-03-14_14-02](/pic/2022-03-14_14-02.png)

#### Incremental Srtucture-from-Motion

![2022-03-14_14-03](/pic/2022-03-14_14-03.png)

##### Feature Extraction

![2022-03-14_14-04](/pic/2022-03-14_14-04.png)

##### Feature Matching & Geometric Veriﬁcation

![2022-03-14_14-05](/pic/2022-03-14_14-05.png)

##### Initialization

![2022-03-14_14-06](/pic/2022-03-14_14-06.png)

##### Image Registration

![2022-03-14_14-16](/pic/2022-03-14_14-16.png)

##### Triangulation

![2022-03-14_14-17](/pic/2022-03-14_14-17.png)

##### Bundle Adjustment & Outlier Filtering

![2022-03-14_14-18](/pic/2022-03-14_14-18.png)

### Results and Applications

![2022-03-14_14-19](/pic/2022-03-14_14-19.png)

![2022-03-14_14-19_1](/pic/2022-03-14_14-19_1.png)

![2022-03-14_14-19_2](/pic/2022-03-14_14-19_2.png)

![2022-03-14_14-20](/pic/2022-03-14_14-20.png)

![2022-03-14_14-20_1](/pic/2022-03-14_14-20_1.png)

![2022-03-14_14-20_2](/pic/2022-03-14_14-20_2.png)

## L4 Stereo Reconstruction

### 4.1 Preliminaries

#### How to recover 3D from an image?

![2022-03-14_14-23](/pic/2022-03-14_14-23.png)

![2022-03-14_14-25](/pic/2022-03-14_14-25.png)

![2022-03-14_14-23_2](/pic/2022-03-14_14-23_2.png)

![2022-03-14_14-24](/pic/2022-03-14_14-24.png)

![2022-03-14_14-24_1](/pic/2022-03-14_14-24_1.png)

#### Why Binocular Stereopsis?

![2022-03-14_14-26](/pic/2022-03-14_14-26.png)

![2022-03-14_14-27](/pic/2022-03-14_14-27.png)

#### Two-View Stereo Matching

![2022-03-14_14-27_1](/pic/2022-03-14_14-27_1.png)

![2022-03-14_14-27_2](/pic/2022-03-14_14-27_2.png)

#### 3D Reconstruction Pipeline

![2022-03-14_14-28](/pic/2022-03-14_14-28.png)

#### 3D Model

![2022-03-14_14-28_1](/pic/2022-03-14_14-28_1.png)

#### Epipolar Geometry

![2022-03-14_14-31](/pic/2022-03-14_14-31.png)

![2022-03-14_14-31_1](/pic/2022-03-14_14-31_1.png)

#### Image Rectification

![2022-03-14_14-34](/pic/2022-03-14_14-34.png)

![2022-03-14_14-34_1](/pic/2022-03-14_14-34_1.png)

![2022-03-14_14-34_2](/pic/2022-03-14_14-34_2.png)

![2022-03-14_14-34_3](/pic/2022-03-14_14-34_3.png)

![2022-03-14_14-34_4](/pic/2022-03-14_14-34_4.png)

#### Calculating the Rectifying Rotation Matrix

![2022-03-14_14-38](/pic/2022-03-14_14-38.png)

#### Rectification Example

![2022-03-14_14-39](/pic/2022-03-14_14-39.png)

#### Disparity Estimation Example

**Left**

![2022-03-14_14-41](/pic/2022-03-14_14-41.png)

**Right**

![2022-03-14_14-41_1](/pic/2022-03-14_14-41_1.png)

![2022-03-14_14-41_2](/pic/2022-03-14_14-41_2.png)

#### Disparity to Depth

![2022-03-14_14-42](/pic/2022-03-14_14-42.png)

### 4.2 Block Matching

#### Correspondence Ambiguity

![2022-03-14_14-43](/pic/2022-03-14_14-43.png)

#### Similarity Metrics

**Left**

![2022-03-14_14-44](/pic/2022-03-14_14-44.png)

**Right**

![2022-03-14_14-46](/pic/2022-03-14_14-46.png)

![2022-03-14_14-46_1](/pic/2022-03-14_14-46_1.png)

![2022-03-14_14-46_2](/pic/2022-03-14_14-46_2.png)

![2022-03-14_14-47](/pic/2022-03-14_14-47.png)

#### Block Matching

![2022-03-14_14-47_1](/pic/2022-03-14_14-47_1.png)

##### Block Matching: Half Occlusions

![2022-03-14_14-48](/pic/2022-03-14_14-48.png)

![2022-03-14_14-48_1](/pic/2022-03-14_14-48_1.png)

##### Block Matching: Assumption Violations

**Left**

![2022-03-14_14-49](/pic/2022-03-14_14-49.png)

**Right**

![2022-03-14_14-49_1](/pic/2022-03-14_14-49_1.png)

**Left**

![2022-03-14_14-51](/pic/2022-03-14_14-51.png)

**Right**

![2022-03-14_14-51_1](/pic/2022-03-14_14-51_1.png)

##### Effect of Window Size

![2022-03-14_14-52_1](/pic/2022-03-14_14-52_1.png)

![2022-03-14_14-52](/pic/2022-03-14_14-52.png)

![2022-03-14_14-53](/pic/2022-03-14_14-53.png)

#### Left-Right Consistency Test

![2022-03-14_14-53_1](/pic/2022-03-14_14-53_1.png)

### 4.3 Siamese Networks

#### Siamese Networks for Stereo Matching

![2022-03-14_14-54](/pic/2022-03-14_14-54.png)

![2022-03-14_14-54_1](/pic/2022-03-14_14-54_1.png)

![2022-03-14_14-54_2](/pic/2022-03-14_14-54_2.png)

#### Training

![2022-03-14_14-55](/pic/2022-03-14_14-55.png)

![2022-03-14_14-55_1](/pic/2022-03-14_14-55_1.png)

#### Loss Function

![2022-03-14_14-56](/pic/2022-03-14_14-56.png)

![2022-03-14_14-56_1](/pic/2022-03-14_14-56_1.png)

#### Winner-takes-All Results

![2022-03-14_14-56_2](/pic/2022-03-14_14-56_2.png)

#### Semiglobal Matching Results

![2022-03-14_14-57](/pic/2022-03-14_14-57.png)

#### Runtime

![2022-03-14_14-57_1](/pic/2022-03-14_14-57_1.png)

### 4.4 Spatial Regularization

#### When will local matching fail?

##### The Underlying Assumption

![2022-03-14_14-58](/pic/2022-03-14_14-58.png)

##### Similarity Constraint: Failure Cases

![2022-03-14_14-58_1](/pic/2022-03-14_14-58_1.png)

#### Spatial Regularization

##### How does the real world look like?

![2022-03-14_14-59](/pic/2022-03-14_14-59.png)

##### Stereo MRF

![2022-03-14_15-00](/pic/2022-03-14_15-00.png)

##### Results

Huang, Lee and Mumford: Statistics of Range Images. CVPR, 2000.

![2022-03-14_15-00_1](/pic/2022-03-14_15-00_1.png)

Güney and Geiger: Displets: Resolving stereo ambiguities using object knowledge. CVPR, 2015.

![2022-03-14_15-00_2](/pic/2022-03-14_15-00_2.png)

### 4.5 End-to-End Learning

#### DispNet

![2022-03-14_15-10](/pic/2022-03-14_15-10.png)

##### Synthetic Datasets

![2022-03-14_15-11](/pic/2022-03-14_15-11.png)

##### DispNet Results on KITTI Dataset

![2022-03-14_15-14](/pic/2022-03-14_15-14.png)

#### GC-Net

![2022-03-14_15-14_1](/pic/2022-03-14_15-14_1.png)

#### Stereo Mixture Density Networks (SMD-Nets)

![2022-03-14_15-15](/pic/2022-03-14_15-15.png)

![2022-03-14_15-15_1](/pic/2022-03-14_15-15_1.png)

![2022-03-14_15-16](/pic/2022-03-14_15-16.png)

![2022-03-14_15-16_1](/pic/2022-03-14_15-16_1.png)

![2022-03-14_15-16_2](/pic/2022-03-14_15-16_2.png)

## L5 Probabilistic Graphical Models

### 5.1 Structured Prediction

#### Block Matching Ambiguities

![2022-03-14_15-22](/pic/2022-03-14_15-22.png)

![2022-03-14_15-22_1](/pic/2022-03-14_15-22_1.png)

#### How does the real world look like?

![2022-03-14_15-22_2](/pic/2022-03-14_15-22_2.png)

#### Spatial Regularization

![2022-03-14_15-23](/pic/2022-03-14_15-23.png)

#### Probabilistic Graphical Models

![2022-03-14_15-23_1](/pic/2022-03-14_15-23_1.png)

#### Structured Prediction

![2022-03-14_15-24](/pic/2022-03-14_15-24.png)

##### Supervised Learning

![2022-03-14_15-24_1](/pic/2022-03-14_15-24_1.png)

###### Classiﬁcation / Regression

![2022-03-14_15-24_2](/pic/2022-03-14_15-24_2.png)

###### Structured Prediction

![2022-03-14_15-25](/pic/2022-03-14_15-25.png)

#### Probabilistic Graphical Models

![2022-03-14_15-27](/pic/2022-03-14_15-27.png)

#### Overview

![2022-03-14_15-28](/pic/2022-03-14_15-28.png)

http://www.nowozin.net/sebastian/cvpr2012tutorial/

http://www.cs.ucl.ac.uk/staff/d.barber/brml/

### 5.2 Markov Random Fields

#### Probability Theory Recap

![2022-03-14_15-30](/pic/2022-03-14_15-30.png)

#### Markov Random Field

![2022-03-14_15-30_1](/pic/2022-03-14_15-30_1.png)

#### Undirected Graph

![2022-03-14_15-31](/pic/2022-03-14_15-31.png)

#### Properties of Markov Random Fields

![2022-03-14_15-31_1](/pic/2022-03-14_15-31_1.png)

![2022-03-14_15-31_2](/pic/2022-03-14_15-31_2.png)

![2022-03-14_15-32](/pic/2022-03-14_15-32.png)

![2022-03-14_15-32_1](/pic/2022-03-14_15-32_1.png)

![2022-03-14_15-32_2](/pic/2022-03-14_15-32_2.png)

#### Global Markov Property

![2022-03-14_15-32_3](/pic/2022-03-14_15-32_3.png)

#### Local Markov Property

![2022-03-14_15-38](/pic/2022-03-14_15-38.png)

##### Example

![2022-03-14_15-38_1](/pic/2022-03-14_15-38_1.png)

#### Hammersley-Clifford Theorem

![2022-03-14_15-40](/pic/2022-03-14_15-40.png)

### 5.3 Factor Graphs

#### MRF Factorization Ambiguities

![2022-03-14_15-41](/pic/2022-03-14_15-41.png)

#### Factor Graphs

![2022-03-14_15-41_1](/pic/2022-03-14_15-41_1.png)

![2022-03-14_15-42](/pic/2022-03-14_15-42.png)

##### Examples

![2022-03-14_15-42_1](/pic/2022-03-14_15-42_1.png)

![2022-03-14_15-42_2](/pic/2022-03-14_15-42_2.png)

### 5.4 Belief Propagation

#### Inference in Chain Structured Factor Graphs

![2022-03-14_15-43](/pic/2022-03-14_15-43.png)

![2022-03-14_15-43_1](/pic/2022-03-14_15-43_1.png)

![2022-03-14_15-43_2](/pic/2022-03-14_15-43_2.png)

#### Inference in Tree Structured Factor Graphs

![2022-03-14_15-43_3](/pic/2022-03-14_15-43_3.png)

![2022-03-14_15-44](/pic/2022-03-14_15-44.png)

![2022-03-14_15-44_1](/pic/2022-03-14_15-44_1.png)

![2022-03-14_15-45](/pic/2022-03-14_15-45.png)

![2022-03-14_15-45_1](/pic/2022-03-14_15-45_1.png)

#### Factor-to-Variable Messages

![2022-03-14_15-46](/pic/2022-03-14_15-46.png)

#### Variable-to-Factor Messages

![2022-03-14_15-46_1](/pic/2022-03-14_15-46_1.png)

![2022-03-14_15-46_2](/pic/2022-03-14_15-46_2.png)

![2022-03-14_15-47](/pic/2022-03-14_15-47.png)

![2022-03-14_15-47_1](/pic/2022-03-14_15-47_1.png)

#### Comments

![2022-03-14_15-47_2](/pic/2022-03-14_15-47_2.png)

#### Sum-Product Algorithm

![2022-03-14_15-48](/pic/2022-03-14_15-48.png)

##### 1. Initialization

![2022-03-14_15-48_1](/pic/2022-03-14_15-48_1.png)

##### 2. Variable-to-Factor Message

![2022-03-14_15-49](/pic/2022-03-14_15-49.png)

##### 3. Factor-to-Variable Message (Sum-Product)

![2022-03-14_15-50](/pic/2022-03-14_15-50.png)

##### 5. Calculate Marginals

![2022-03-14_15-50_1](/pic/2022-03-14_15-50_1.png)

#### Log Representation

![2022-03-14_15-51](/pic/2022-03-14_15-51.png)

![2022-03-14_15-51_1](/pic/2022-03-14_15-51_1.png)

#### Max-Product Algorithm

##### Finding the maximal state: Max-Product

![2022-03-14_15-52](/pic/2022-03-14_15-52.png)

##### Example: Chain

![2022-03-14_15-52_1](/pic/2022-03-14_15-52_1.png)

![2022-03-14_15-53](/pic/2022-03-14_15-53.png)

##### Max-Product Algorithm – Overview

![2022-03-14_15-53_1](/pic/2022-03-14_15-53_1.png)

#### Loopy Belief Propagation

![2022-03-14_15-53_2](/pic/2022-03-14_15-53_2.png)

![2022-03-14_15-54](/pic/2022-03-14_15-54.png)

#### Summary

##### Sum-Product Belief Propagation 

![2022-03-14_15-55](/pic/2022-03-14_15-55.png)

##### Max-Product Belief Propagation

![2022-03-14_15-55_1](/pic/2022-03-14_15-55_1.png)

##### Special Case: Pairwise MRF

![2022-03-14_15-56](/pic/2022-03-14_15-56.png)

![2022-03-14_15-56_1](/pic/2022-03-14_15-56_1.png)

##### Readout

![2022-03-14_15-56_2](/pic/2022-03-14_15-56_2.png)

##### Algorithm Overview

![2022-03-14_15-57](/pic/2022-03-14_15-57.png)

### 5.5 Examples

#### Example 1: Vehicle Localization

![2022-03-14_15-57_1](/pic/2022-03-14_15-57_1.png)

![2022-03-14_15-58](/pic/2022-03-14_15-58.png)

![2022-03-14_15-58_1](/pic/2022-03-14_15-58_1.png)

![2022-03-14_15-58_2](/pic/2022-03-14_15-58_2.png)

![2022-03-14_15-59](/pic/2022-03-14_15-59.png)

![2022-03-14_15-59_1](/pic/2022-03-14_15-59_1.png)

![2022-03-14_15-59_2](/pic/2022-03-14_15-59_2.png)

#### Example 2: Image Denoising

![2022-03-14_16-00](/pic/2022-03-14_16-00.png)

![2022-03-14_16-00_1](/pic/2022-03-14_16-00_1.png)

![2022-03-14_16-00_2](/pic/2022-03-14_16-00_2.png)

![2022-03-14_16-00_3](/pic/2022-03-14_16-00_3.png)

![2022-03-14_16-01](/pic/2022-03-14_16-01.png)

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
