# Classification Model Comparison

## Introduction

This classification task evaluates three deep learning models for binary breast ultrasound image classification into **malignant** and **benign** classes. The compared models are **DenseNet121**, **VGG16**, and **ResNet50**, all initialized with ImageNet-pretrained weights and adapted for a two-class output layer.

DenseNet121 is a densely connected convolutional neural network where each layer receives feature maps from earlier layers. This design supports strong feature reuse and can help the model learn detailed image patterns with fewer redundant parameters. VGG16 is a classic convolutional network with a simple stack of convolutional and pooling layers, making it a useful baseline architecture for image classification. ResNet50 uses residual connections, which help deeper networks train more effectively by allowing information to pass through skip connections.

## Experimental Settings

The dataset was split into a training set and a test set. The training set contained **517 images**: 171 malignant and 346 benign. The test set contained **130 images**: 39 malignant and 91 benign. All images were resized to **224 x 224** pixels before being passed into the models.

The models were trained in PyTorch using ImageNet normalization. Training data augmentation included random horizontal flipping, random rotation, and slight color jittering. A weighted sampling strategy and class-weighted cross-entropy loss were used to reduce the effect of class imbalance. The optimizer was **AdamW** with weight decay. Each model was trained for up to **80 epochs**, beginning with the classifier head and then fine-tuning later backbone layers after the first 5 epochs. Early stopping was used with a patience of 15 epochs. Final classification performance was evaluated on the shared test set, and ROC AUC was calculated using the benign class as the positive class.

## Numeric Results

The table below summarizes the test-set performance. Precision, recall, and F1 score are reported as macro averages across the malignant and benign classes.

| Model | Accuracy | Recall | Precision | F1 Score | AUC |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 0.8692 | 0.8480 | 0.8431 | 0.8454 | 0.9110 |
| VGG16 | 0.9000 | 0.8773 | 0.8830 | 0.8801 | 0.8938 |
| ResNet50 | 0.8769 | 0.8608 | 0.8509 | 0.8556 | 0.9169 |

## Findings

VGG16 achieved the best overall classification performance by accuracy and macro F1 score. It reached an accuracy of **0.9000** and a macro F1 score of **0.8801**, meaning it produced the strongest balance between precision and recall on the final class predictions.

ResNet50 achieved the highest AUC value, with an AUC of **0.9169**. This suggests that ResNet50 had the strongest overall ability to separate benign and malignant cases across different probability thresholds, even though its final threshold-based accuracy and F1 score were slightly lower than VGG16.

DenseNet121 also performed well, with an AUC of **0.9110** and accuracy of **0.8692**. Its results were close to ResNet50, showing that it learned useful discriminative features, but its macro F1 score was the lowest among the three models.

Overall, all three models showed strong performance, with AUC values close to or above 0.90. This indicates that the models were able to distinguish between benign and malignant ultrasound images much better than random classification.

## Conclusion

The comparison shows that **VGG16** was the best model for final test-set classification because it achieved the highest accuracy and macro F1 score. However, **ResNet50** produced the highest AUC, making it the strongest model in terms of ranking and class separation across thresholds. **DenseNet121** remained competitive and performed especially well in ROC analysis. Based on these results, VGG16 is the best choice when the goal is highest final classification accuracy, while ResNet50 may be preferred when threshold flexibility and ROC performance are more important.
