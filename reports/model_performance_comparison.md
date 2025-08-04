# Model Performance Comparison

This document compares the performance metrics of the fine-tuned classifier against the Week 1 baseline models (Logistic Regression and SVM). The classifier was designed to predict whether text is AI-generated or not.

## Baseline Model Performances

### Logistic Regression Model
- **Accuracy:** 0.9949  
- **F1 Score:** 0.9931  
- **Precision:** 0.9969  
- **Recall:** 0.9893  

### SVM Model
- **Accuracy:** 0.9988  
- **F1 Score:** 0.9984  
- **Precision:** 0.9996  
- **Recall:** 0.9972  

## Fine-Tuned Model Performance

- **Accuracy:** 0.9997  
- **F1 Score:** 0.9996  
- **Precision:** 0.9996  
- **Recall:** 0.9997  

## Comparison Analysis

### Against Logistic Regression:
- **Accuracy:** Improved by ≈ 0.48% (from 99.49% to 99.97%)
- **F1 Score:** Improved by ≈ 0.65% (from 99.31% to 99.96%)
- **Precision:** Improved by ≈ 0.27% (from 99.69% to 99.96%)
- **Recall:** Improved by ≈ 1.03% (from 98.93% to 99.97%)

### Against SVM:
- **Accuracy:** Improved by ≈ 0.09% (from 99.88% to 99.97%)
- **F1 Score:** Improved by ≈ 0.12% (from 99.84% to 99.96%)
- **Precision:** Nearly identical (99.96% vs 99.96%)
- **Recall:** Improved by ≈ 0.25% (from 99.72% to 99.97%)

## Conclusion

The fine-tuned model outperforms both baseline models. The improvements are most significant when compared to the Logistic Regression model, while gains over the SVM model are more modest. Overall, the enhancements reflect a more robust classifier with slightly higher accuracy, F1 score, and recall.

The fine-tuned model benefits from advanced training techniques and additional task-specific data, which help it capture subtle linguistic patterns better than the generic baseline models. This targeted refinement leads to improved parameter tuning and overall performance.
