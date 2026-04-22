# evolution algorithm:

## the project already chosen:

### [3] Feature Selection in Image Analysis using Genetic Algorithms [Preferably in combination with other EAs/SI approaches of your choice]:

- o Identify a publicly available imaging dataset (e.g., for diagnosing diseases).
- o Define an objective function for feature selection that maximises classification accuracy while reducing feature count.
- o Implement a Genetic Algorithm (GA) where each individual represents a feature subset.
- o Run experiments comparing different selection and crossover strategies on the dataset.
- o Summarise findings with charts, accuracy metrics, and a final report.
- o You may employ other EAs/SI approaches instead of GAs.

#### Detailed Description:

Context and Problem Statement: In image analysis, reducing the number of features (or image characteristics) can simplify models and improve performance. Selecting the best features is critical for accurate diagnosis.

#### Key Terms and Concepts:

- o Feature Selection: The process of choosing a subset of relevant features (variables) for model training.
- o Genetic Algorithm (GA): An algorithm that evolves candidate solutions using techniques inspired by biological evolution.
- o Objective Function: In this case, it balances high accuracy and lower complexity.

#### Requirements/Deliverables:

- o A GA implementation for selecting a meaningful subset of image features.
- o A detailed comparison of different GA operators (selection, crossover, mutation).
- o Performance metrics (accuracy, reduction percentage) across multiple runs.
- o A comprehensive report with visual aids (graphs and tables) showing algorithm performance.


# general instruction in this course project:

## Instructions to Students:

- o This is a group work project. Each group consists of five to six students (the Teaching Assistant must approve group members through registration). Each group must develop the assigned idea using Python.
- o Project Objectives: The objectives of this project can be summarised as applying the main ideas, fundamental concepts, and basic algorithms in the field of Evolutionary Algorithms & Computing.
- o Submission: Submission is done according to the following schedule:
- o Week 13/14: Submission and Discussion of the (1) Project and (2) Documentation. The report should include the following: (1) Project idea in detail, (2) Main functionalities, (3) Similar applications in the market, (4) A literature review of Academic publications (papers) relevant to the idea (at least 4 to 6 papers, as per the number of team members), (5) the Dataset employed (preferably a publicly available dataset), (6) Details of the algorithm(s)/approach(es) used and the results of the experiments, and (7) Development platform.
- o Assessment: Assessment will be on the reports, code submitted, and discussions with team members. All team members must contribute across all phases, and each member's role must be clearly stated in each report.
- o The Project will be assessed based on the following criteria:
  - ▪ The complexity of the problem, & the correctness of the algorithms employed.
  - ▪ The quality/comprehensiveness of your experiments & documentation.
  - ▪ The correctness of your analysis and design diagrams.
  - ▪ Implementation correctness.
- o Feedback: If requested, further details and feedback could be provided for each group through discussions with the teaching assistant(s) during the weekly labs/office hours.
- o You can only submit your work. Any student suspected of plagiarism will be subject to the procedures set out by the Faculty/University (including failure of the course).
- o Academic Integrity: The University’s policies on academic integrity will be enforced against students who violate the University's standards of academic integrity. Examples of behaviour that is not allowed are:
  - ▪ Copying all or part of someone else's work and submitting it as your own;
  - ▪ Giving another student in the class a copy of your work and
  - ▪ Copying parts from the internet, textbooks, etc.
  - ▪ If you have any questions concerning what is allowed, please don’t hesitate to discuss them with me.
- o N.B., We understand that you might positively refer to AI tools to support your research. Please note that, in case there are any concerns about AI-generated submission content, you will be invited for an opportunity to verify and defend your work.

## and:

### Important Guidelines for ALL Projects to follow:

a) Clearly define and formalise your problem as one of the problem types we’ve studied throughout this module: Optimisation, Modelling, Simulation, Constraint Satisfaction, Free Optimisation, Constrained Optimisation, etc.
b) If your selected idea mandates Constraint Handling, clearly implement an approach to handling constraints (i.e., Penalty Functions, Repair Functions, Restricting Search to the Feasible Region, Decoder Functions, etc.).
c) If your selected idea mandates Coevolution, clearly implement a coevolutionary approach (cooperative or competitive).
d) Clearly define the Components of your Evolutionary Algorithm: For instance, in the case of a genetic algorithm, clearly define the Representation (Definition of Individuals), Evaluation Function (Fitness Function), Population, Parent Selection Mechanism, Variation Operators (Mutation and Recombination), Survivor Selection Mechanism (Replacement), Initialisation, and Termination Condition(s).
e) Select variation operators (mutation and recombination) suitable for the selected representation.
   - • If possible, use at least 2 parent selection techniques (each independently) and report the results for each.
   - • If possible, use at least 2 recombination techniques (each independently) and report the results for each.
   - • If possible, use at least 2 mutation techniques (each independently) and report the results for each.
f) If possible, use at least 2 population-management-models/survivor-selection (each independently) and report the results for each.
g) Clearly describe and implement approaches to control/tune the parameters.
h) Describe and implement a suitable approach for preserving diversity (i.e., Fitness Sharing, Crowding, Automatic Speciation Using Mating Restrictions, Running Multiple Populations in Tandem, such as the Island Model EAs, Spatial Distribution within One Population, such as Cellular EAs, etc.).
i) Incorporate a functional user interface demonstrating the algorithm, parameters, inputs, and results.
j) The students may be awarded bonus marks in the following cases (only if the experiments are carried out properly, and the results/performance were measured, reported, and analysed adequately:
   - • Investigating the effect of multiple (at least 2) representations (when possible).
   - • Investigating the effect of multiple (at least 2) initialisation approaches (when possible).
   - • Investigating the effect of over-selection for large populations (when possible).
   - • An educational visual interface that illustrates (simulates) the changes in the evolutionary process and solutions when different parameters/options/approaches are selected. I.e., an interface/simulation that can be later used to teach students the effects of varying selected approaches/parameters, etc.
   - • Contributions (experiments and results) with high publication potential.
   - • Hybrid approaches (employing more than a single EAs/SI approach).
   - • Employing SOTA novel variants of the EAs/SI approaches rather than the traditional implementations.

------------------------------------------------------

# computer vision:

## chosen project:

### 4) Traffic Sign Detection and Recognition

Project Idea: This project aims to build a system capable of detecting and recognizing traffic signs from road images or video streams. The system first applies image preprocessing techniques to improve visibility, then uses object detection methods to locate traffic signs in the scene. After detection, the system extracts features and classifies each sign into predefined categories such as stop, speed limit, or warning signs.

Expected Outcome: An intelligent system that can understand road signs and support autonomous driving applications.

## project general reqs:

### Suggested Project Requirements

#### Core Functional Requirements (Mandatory)

1. Image Preprocessing Module
   -  Apply at least 2 filtering techniques
     - o Gaussian Filter
     - o Median Filter
   -  Compare results visually and numerically
2. Feature Detection Module
   -  Implement Harris Corner Detector
   -  Display detected corners on images
   -  Analyze effect of threshold tuning
3. Multi-Scale Analysis Module
   -  Implement Gaussian/Laplacian Pyramid
   -  Demonstrate scale-space analysis
   -  Show object detection at multiple scales
4. Feature Extraction & Matching Module
   -  Implement/use SIFT
   -  Match keypoints between image pairs
   -  Draw match visualization
5. Image Segmentation Module
   -  Segment main objects/regions from image
   -  Can use:
     - o Thresholding
     - o Region Growing
     - o Watershed
     - o K-means segmentation
6. Classification Module
   -  Train Naive Bayes OR Boosting classifier
   -  Classify segmented objects/regions
7. Final Integrated Pipeline
   Combine all modules into one working application.

#### Constraints:

Dataset
-  Must use public dataset OR collect own dataset
-  Minimum dataset size: 200 images

Evaluation
-  Must provide quantitative metrics:
  - o Accuracy
  - o Precision/Recall
  - o IoU for segmentation
  - o Matching accuracy

#### Grading Rubric (20 Marks)

| Criterion | Marks |
| --- | --- |
| Filtering Implementation | 2 |
| Harris Detector | 2 |
| Pyramid Implementation | 1 |
| SIFT Implementation | 3 |
| Segmentation Quality | 2 |
| Classification Accuracy | 2 |
| Integration / Final Pipeline | 2 |
| Report Quality | 1 |

---------------------------------------------------------

# AML:

## general guidline:

### General Guidelines for All Projects

#### Required Technology Stack

- PyTorch + TorchVision - Training and transfer learning
- FastAPI + Uvicorn - REST API deployment
- Docker - Containerized deployment
- Pillow - Image preprocessing
- scikit-learn - Metrics (confusion matrix, classification report)

#### Installation

```sh
pip install torch torchvision fastapi uvicorn python-multipart Pillow scikit-learn
```

#### Standard Team Roles (7 Members)

# Role Tasks

- 1 Data Manager Download dataset, organize train/val/test folders, class balance analysis
- 2 EDA & Visualizer Sample grids, class distributions, image properties, 8+ plots
- 3 Augmentation Designer Design augmentation pipeline, compare with/without augmentation
- 4 Model Trainer Transfer learning setup, training loop, loss/accuracy curves
- 5 Evaluator Confusion matrix, per-class metrics, misclassification analysis
- 6 API Developer FastAPI /predict endpoint, Swagger docs, preprocessing pipeline
- 7 Deployer & Presenter Dockerfile, test script, demo slides, live presentation

#### Timeline (4 Weeks)

Week Milestone Responsible

- 1 Dataset download, EDA notebook, augmentation Roles 1-3
- 2 Train model, export .pth, evaluate Roles 3-5
- 3 Build FastAPI, Dockerfile, test script Roles 5-7
- 4 Integration test, presentation prep, live demo All roles

#### Required Deliverables (All Projects)

- 1. EDA Notebook - 8+ visualizations with written insights
- 2. Trained Model - Exported as .pth file
- 3. Training Report - Loss/accuracy curves, best epoch, final metrics
- 4. Evaluation - Confusion matrix, per-class accuracy, F1 scores
- 5. FastAPI App - /predict endpoint accepting image uploads
- 6. Dockerfile - Containerized deployment ready
- 7. Test Script - Automated API testing (test_api.py)
- 8. Presentation - 10-minute demo with live API test

Page 2/23
Advanced ML Team Project Guidelines

#### FastAPI Template (All Projects)

```python
# app.py
from fastapi import FastAPI, UploadFile, File
import torch, io
from PIL import Image
from torchvision import transforms, models
app = FastAPI(title="[Project Name] API")
model = load_your_model("model.pth") # Load trained model
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
img = Image.open(io.BytesIO(await file.read())).convert("RGB")
tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
probs = torch.softmax(model(tensor), dim=1)[0]
return {"class": CLASSES[probs.argmax()],
"confidence": float(probs.max())}
# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

Page


## the project:

### Project 3: Traffic Sign Recognition

Goal: Classify traffic sign images into 43 categories for driver assistance systems.

#### Project Details

- Dataset GTSRB - German Traffic Sign Recognition
- Kaggle URL https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign-recognition-benchmark
- Classes 43 (speed limits, stop, yield, no entry, etc.)
- Total Images 50,000+
- Model MobileNetV2 or EfficientNet-B0 (transfer learning)
- Export File traffic_sign_model.pth
- Target >90% overall accuracy

#### Dataset Description

50,000+ images of 43 different German traffic signs. Images vary in size (15x15 to 250x250). Includes speed limits, stop
signs, yield, no entry, and more. Very imbalanced classes.

#### API Response Example

```json
{"sign": "Speed limit (30km/h)", "category": "prohibition", "confidence": 0.97}
```

#### Business Value

Self-driving car assistance, driver safety apps, traffic monitoring.

#### Key Challenges

- Varying image sizes - need consistent resizing
- Very imbalanced (some signs have 200 images, others 2000)
- Similar looking signs - speed limit 30 vs 50 vs 80

#### Tips for Success

- Resize all to 224x224 for transfer learning
- Use brightness/contrast jitter (signs appear in sun and shadow)
- Test with real photos taken with phone camera

#### Team Roles

# Role Tasks

- 1 Data Manager Download, resize to 224x224, create balanced train/val/test split
- 2 EDA & Visualizer Class distribution (imbalanced!), sample grids, similar sign pairs
- 3 Augmentation Brightness/contrast jitter, rotation, handle rare classes
- 4 Model Trainer Transfer learning, experiment with 2+ architectures, pick best
- 5 Evaluator Per-class accuracy, confusion matrix, top-5 accuracy, similar sign analysis

Page 8/23
Advanced ML Team Project Guidelines

- 6 API Developer FastAPI with sign name + category (warning/prohibition) + top-3
- 7 Deployer Docker, test with phone photos, demo, presentation


# cloud:

mentioned in cloud_proposal.md