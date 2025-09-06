# Hemo_Scan

Hemo_Scan is an AI-powered project that detects anemia risk from fingernail images.  
It uses deep learning to analyze nail photos and give a quick, non-invasive health check.

Inspiration
Anemia is one of the most common health problems worldwide, but many people don’t even know they have it.  
We wanted to build an **easy and accessible solution** that helps people check anemia risk without expensive tests.

What it does
- Takes a fingernail photo as input.  
- Analyzes the image with AI (MobileNet-based model).  
- Predicts the anemia class: **Anemic, Borderline, or Normal**.  
- Estimates Hemoglobin (Hb) level.  
- Shows a risk message (High, Medium, Low).  
- Provides a heatmap (Grad-CAM) to highlight the important nail areas.  

How we built it
- Collected dataset of nail images with hemoglobin levels.  
- Preprocessed data using **OpenCV & Pandas**.  
- Trained deep learning model with **PyTorch** (MobileNet backbone).  
- Balanced classes with weighted sampling and data augmentation.  
- Tested on validation images.  
- Created Grad-CAM overlays for explainability.  

Challenges
- Small and imbalanced dataset.  
- Difficult to extract clear features from nails.  
- Model accuracy is still improving (work in progress).

 Accomplishments
- Built a working ML pipeline end-to-end.  
- Designed a dual-head model (classification + Hb regression).  
- Achieved useful predictions that can guide further improvements.  
- Created an interpretable AI with Grad-CAM visualization.  

 What we learned
- How to handle imbalanced medical datasets.  
- How to combine classification + regression tasks in one model.  
- The importance of explainability in healthcare AI.  

What's next
- Collect larger and more diverse dataset.  
- Improve accuracy and reduce bias.  
- Build a **web app / mobile app** for real-world use.  
- Collaborate with healthcare professionals for validation.  

 

