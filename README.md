# Plant Disease Classification Analysis

This project analyzes a dataset of plant leaf images to classify various species and disease conditions. The workflow progresses from Exploratory Data Analysis (EDA) to implementing Convolutional Neural Networks and Transfer Learning.

## Exploratory Data Analysis

We analyzed the dataset to understand class distributions and image quality before training. The dataset is divided into three categories: Color, Grayscale, and Segmented.

### Class Distribution
The dataset contains 38 classes with a total distribution shown below. There is a significant class imbalance, with **Orange___Haunglongbing** having the most samples and **Potato___healthy** having the fewest.

**Top 5 Classes:**
- Orange___Haunglongbing_(Citrus_greening): 5507
- Tomato___Tomato_Yellow_Leaf_Curl_Virus: 5357
- Soybean___healthy: 5090
- Peach___Bacterial_spot: 2297
- Tomato___Bacterial_spot: 2127

**Bottom 5 Classes:**
- Raspberry___healthy: 371
- Peach___healthy: 360
- Apple___Cedar_apple_rust: 275
- Potato___healthy: 152

![Class Frequency Plot](assets/class_frequencies.png)

This imbalance (ratio of approx. 36:1) suggests the need for augmentation or class weighting to prevent bias towards the majority classes.

### Image Specifications
All images across the three folders (Color, Grayscale, Segmented) are standardized to the same dimensions.
- **Dimensions:** 256x256 pixels
- **Color:** 3 channels (RGB)
- **Grayscale:** 1 channel
- **Segmented:** 3 channels (masked background)

### Blur Detection (Laplacian Variance)
To filter out low-quality data, we calculated a blur score using the variance of the Laplacian. This method utilizes a kernel to approximate the second derivative of the image, where high variance corresponds to sharp edges and low variance corresponds to blurring.

The Laplacian $L$ of an image $I$ is calculated via convolution with a kernel $K$:

$$L(x,y) = I(x,y) * K$$

Where the kernel $K$ is typically:

$$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

The sharpness score $S$ is the variance of the response:

$$S = \frac{1}{N} \sum_{i=1}^{N} (L_i - \mu_L)^2$$

### Blur Report
We computed the average sharpness scores for each dataset category.

![Blur Sharpness Plot](assets/sharpness_plot.png)

**Dataset Statistics:**
- **Global Pixel Mean:** 0.52
- **Global Pixel Std:** 0.15

**Average Laplacian Variance (Sharpness):**
- **Color:** 2483.42
- **Grayscale:** 1825.19
- **Segmented:** 2875.98

The segmented images display the highest variance (2875.98) due to the high-contrast artificial edges created between the leaf and the black background. Grayscale images have the lowest variance (1825.19) as the conversion reduces the high-frequency information present in color channels.

## Shallow Models Baseline

To establish a baseline before deep learning, we evaluated three classical machine learning approaches.

**Random Forest Classifier (RFC)**
This model was used for its robustness to overfitting, employing an ensemble of decision trees to capture non-linear feature interactions without heavy tuning.

**SGD Classifier**
We tested a linear classifier optimized via stochastic gradient descent. This provided a computationally efficient benchmark for handling the high-dimensional flattened image data.

**Hierarchical Model**
We implemented a two-stage classification strategy. The model first predicts the plant species (e.g., Tomato vs. Corn) and subsequently classifies the specific disease within that species, effectively breaking the complex 38-class problem into smaller, manageable tasks.

## Model Architecture and Strategy

To achieve high accuracy and robustness, we implemented a two-stage modeling approach. We began with a custom CNN to understand the feature complexity, followed by a Transfer Learning approach using ResNet18 to achieve production-grade performance.

### Stage 1: Custom CNN (Feature Learning)
We engineered a custom Convolutional Neural Network from the ground up to establish a strong baseline. Unlike generic architectures, this model was specifically tuned to handle the high variance of plant disease patterns without relying on pre-trained weights.

**Architecture Breakdown**
The model consists of a 3-stage hierarchical feature extractor:
- **Block 1 (32 Filters):** Captures low-level primitives such as leaf edges and vein structures.
- **Block 2 (64 Filters):** Aggregates primitives into simple geometric shapes, such as circular lesion outlines.
- **Block 3 (128 Filters):** The deepest layer, responsible for recognizing complex textures (e.g., distinguishing the "fuzzy" texture of mold from the "dry" texture of scorch).

**Regularization and Augmentation**
To prevent overfitting, we employed a Dropout rate of 0.5 in the fully connected layer, forcing the network to learn distributed representations. Crucially, we utilized **Geometric Invariance Augmentation** (RandomRotation and RandomHorizontalFlip). This forced the model to learn features independent of orientation, solving the issue of positional bias common in small datasets and allowing the model to generalize significantly better than standard implementations.

### Stage 2: Transfer Learning (ResNet18)
To push accuracy to **99.6%**, we utilized ResNet18, a deep residual network pretrained on the ImageNet dataset.

**Why ResNet18?**
We selected ResNet18 over deeper variants (like ResNet50) because plant disease features are primarily local and textural. An 18-layer residual network provides sufficient receptive field coverage without the computational overhead or risk of overfitting associated with deeper models. The residual connections ($y = F(x) + x$) allow gradients to flow through the network without vanishing, enabling the training of deeper feature extractors.

**Implementation Strategy (Discriminative Fine-Tuning)**
We employed a two-step training process to maximize performance:

1. **Head Replacement:** We replaced the final 1000-class ImageNet layer with a 38-class linear layer specific to the PlantVillage classes. The input to this layer is a 512-dimensional feature vector derived from the global average pooling of the final convolutional block.

2. **Fine-Tuning:**
   - Initially, the backbone was frozen to train only the classification head.
   - Subsequently, we unfroze the entire network and applied a significantly lower learning rate (1e-4). This allowed the model to adjust its deep feature representations to the specific domain of plant leaves without catastrophic forgetting of the generalized ImageNet features.

**Data-Centric Optimization**
A key factor in achieving high convergence speed was the use of ImageNet Normalization. By normalizing input images with Mean [0.485, 0.456, 0.406] and Std [0.229, 0.224, 0.225], we aligned our data distribution with the pretrained weights. This ensured that the model's pre-learned filters for edge and color detection remained valid for our specific dataset.
