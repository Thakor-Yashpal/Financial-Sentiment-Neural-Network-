# Financial-Sentiment-Neural-Network-

### Financial Sentiment Analysis using Transformer Models

## Project Overview

This project implements a state-of-the-art financial sentiment analysis system powered by transformer-based models, primarily RoBERTa fine-tuned on a proprietary financial text dataset containing over 5,800 labeled sentences. The model categorizes financial communications into negative, neutral, and positive sentiments to support equity research and investment decision-making.

## One-Page Summary

### Financial Sentiment Analysis Using Transformer-Based Models

**Yashpal Thakor**  
Founder, pyr0mind | Deep Learning Enthusiast  

---

#### Abstract

The project builds a natural language processing system for financial sentiment analysis to extract actionable insights from earnings call transcripts, news, and analyst reports. By fine-tuning RoBERTa on domain-specific financial datasets, the classifier robustly detects sentiment patterns, aiding institutional equity analysis.

#### Methods

A labeled dataset with three sentiment categories was used, maintaining stratified train-validation-test splits. The model uses RoBERTa as a feature extractor supplemented by a multi-layer classification head. Text preprocessing normalizes financial terms, and focal loss with label smoothing addresses class imbalance. Early stopping and learning rate scheduling optimize training.

#### Results

The enhanced RoBERTa model achieved 78.04% accuracy on test data, showing expected improvements over baseline DistilBERT performance. Error analysis highlights strengths in neutral and positive classes with opportunity to improve negative class performance. Training curves demonstrated robust, stable optimization.

#### Conclusion

This implementation provides a scalable architecture for financial sentiment classification that can enhance equity research workflows. Future work includes model ensembling, advanced data augmentation, and adding multimodal financial signals for further accuracy gains.

---

## Repository Contents

- `data.csv` - Financial text dataset with sentiment labels
- `financial_sentiment_model.py` - PyTorch model code with training & evaluation
- `notebooks/` - Jupyter Notebooks for exploration and prototyping
- `results/` - Performance charts, confusion matrices, reports
- `README.md` - This file and project documentation

---

## Getting Started

1. Install the required packages: `pip install -r requirements.txt`
2. Run `financial_sentiment_model.py` to train and evaluate
3. Use the provided prediction functions for inference on new data

---

## Contact

For questions or collaborations, please contact:  
**Yashpalsinh Thakor**  
Email: thakoryashpal755@gmail.com  
GitHub: https://github.com/Thakor-Yashpal

---

kaggle: https://www.kaggle.com/code/yashpalthakor/financial-sentiment-neural-network-01

Kaggle: https://www.kaggle.com/code/yashpalthakor/financial-sentiment-neural-network-02

---

