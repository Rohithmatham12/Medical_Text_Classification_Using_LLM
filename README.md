# Medical Text Classification with LLMs: A Deep Dive into Pathology Report Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Transformers](https://img.shields.io/badge/Transformers-4.47.0-blue)](https://huggingface.co/transformers/)
[![Datasets](https://img.shields.io/badge/Datasets-3.3.1-blue)](https://huggingface.co/datasets/)
[![Accelerate](https://img.shields.io/badge/Accelerate-1.2.1-blue)](https://huggingface.co/accelerate/)

## Overview

This project explores the application of Natural Language Processing (NLP) techniques, specifically Large Language Models (LLMs), for the automated classification of medical pathology reports.  By accurately classifying these reports, we aim to improve diagnostic workflows, accelerate research, and ultimately enhance patient care. This project uses data from The Cancer Genome Atlas (TCGA) project, showcasing my ability to work with real-world medical data and cutting-edge NLP methodologies.

**The core objective:** To develop a robust and accurate multi-class classification model that can automatically identify the cancer type described within a pathology report text.

**Why this is important:**

*   **Improved Efficiency:** Automating the classification of pathology reports can significantly reduce the manual effort required by medical professionals, freeing up their time for more complex tasks.
*   **Enhanced Accuracy:**  AI-powered classification can minimize human error and ensure consistency in diagnostic processes.
*   **Accelerated Research:** Well-categorized pathology reports are crucial for cancer research, enabling faster data retrieval and analysis.
*   **Better Patient Outcomes:** Ultimately, improved diagnostics and research contribute to more effective treatments and better patient outcomes.

## Key Highlights

*   **Comprehensive Preprocessing:** Implemented robust text cleaning and preprocessing steps to handle the complexities of medical language, including handling noise, special characters, and variations in terminology.
*   **Advanced Feature Engineering:** Explored both traditional TF-IDF vectorization and transformer-based embeddings (ClinicalBERT) for feature extraction, comparing their performance in this specific context.
*   **State-of-the-Art LLM Fine-tuning:** Fine-tuned a pre-trained ClinicalBERT model on the TCGA pathology report dataset to achieve high accuracy in cancer type classification.  Also experimented with few-shot learning using FLAN-T5.
*   **Hyperparameter Optimization:**  Leveraged GridSearchCV for rigorous hyperparameter tuning of the TF-IDF + Logistic Regression model, optimizing performance on a held-out validation set.
*   **Performance Analysis:** Conducted in-depth performance analysis, including classification reports, confusion matrices, and ROC curves, with a specific focus on identifying and addressing challenges related to rare cancer types.
*   **Data Augmentation:** Developed and implemented targeted data augmentation techniques to address class imbalance and improve the model's ability to generalize to less frequent cancer types.
*   **Experimentation with Few-Shot Learning:** Explored the potential of FLAN-T5 to tackle the classification task using a limited number of training examples, demonstrating adaptability to resource-constrained scenarios.

## Technical Approach

This project demonstrates a range of NLP techniques and machine learning models. Here's a breakdown of the key components:

1.  **Data Acquisition and Preprocessing:**
    *   Loading and merging pathology report text data with cancer type labels from the TCGA dataset.
    *   Text cleaning: Lowercasing, removing punctuation and special characters, handling extra whitespace.
    *   Label encoding: Converting cancer type strings into numerical labels for model training.
    *   Data splitting:  Dividing the dataset into training (80%), validation (10%), and testing (10%) sets, with stratification to ensure balanced class representation across all sets.

2.  **Traditional Machine Learning Model (TF-IDF + Logistic Regression):**
    *   **TF-IDF Vectorization:**  Converting text reports into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF).  Implemented optimizations:
        *   `ngram_range`: Explored unigrams, bigrams, and trigrams to capture contextual information.
        *   `max_features`: Limiting vocabulary size to reduce dimensionality and improve performance.
        *   `min_df` and `max_df`: Filtering out very rare or very common terms.
        *   `stop_words`: Removal of common English stop words
    *   **Logistic Regression:** Training a Logistic Regression classifier on the TF-IDF vectors to predict cancer types.
    *   **GridSearchCV Optimization:** Using GridSearchCV to find the best hyperparameters for the TF-IDF vectorizer and Logistic Regression classifier, maximizing the weighted F1-score on the validation set.

3.  **Transformer-Based Model (ClinicalBERT Fine-Tuning):**
    *   **ClinicalBERT:** A pre-trained BERT model specifically designed for clinical text.
    *   **Fine-tuning:**  Adapting the ClinicalBERT model to our specific classification task by training it on the TCGA pathology reports.  Key steps:
        *   Tokenization: Using the ClinicalBERT tokenizer to prepare the text for input into the model.
        *   Dataset creation:  Converting the data into PyTorch Dataset objects.
        *   Training: Utilizing the Hugging Face Transformers `Trainer` class to fine-tune ClinicalBERT with optimized training parameters.
    *   **Performance Evaluation:** Evaluating the fine-tuned ClinicalBERT model on the test set using weighted F1-score and detailed classification reports.

4. **Data Augmentation:**
* Using re library for augmentation .
* Used "squamous cell": "epidermoid carcinoma",
"adenocarcinoma": "glandular cancer",
"carcinoma": "malignant neoplasm" rules for augmentation

5.  **Few-Shot Learning with FLAN-T5 (Experimental):**
    *   Loading and using a pre-trained FLAN-T5 model.
    *   Defining a "few-shot" prompt: A template that includes example pathology reports and their corresponding cancer types to guide the model.
    *   Generating predictions for the test set using the few-shot prompt.
    *   Evaluating the F1-score to assess the performance of the few-shot approach.

## Results and Performance

*   **TF-IDF + Logistic Regression:** Achieved a weighted F1-score of `0.965` on the test set. (The complete classification report with precision, recall, and F1-score for each class is available in the notebook output).
*   **ClinicalBERT (Fine-tuned):** Achieved a weighted F1-score of `0.77` on the test set after fine-tuning for just one epoch. Further optimization could likely improve this score.
* Augmented Data : Achieved a weighted F1-score of `0.812` on the test set after fine-tuning for just one epoch.
*   **FLAN-T5 (Few-Shot):** While showing promise, the initial few-shot results were limited, with a low F1-score.  This indicates a need for more sophisticated prompting strategies and further fine-tuning.

**Key observations:**
*  Augmented ClinicalBERT shows improved results comparing to non augmented ones
*   The ClinicalBERT model demonstrated strong potential, even with minimal fine-tuning. Further optimization, including more training epochs, hyperparameter tuning, and more sophisticated handling of class imbalance, could significantly improve its performance.
*   The TF-IDF + Logistic Regression model provided a strong baseline, highlighting the value of even traditional methods when combined with proper feature engineering and optimization.
*   The relatively poor performance of FLAN-T5 in the few-shot setting suggests a promising direction for future research. Developing better prompts and potentially fine-tuning FLAN-T5 could unlock its capabilities for rapid adaptation to new medical text classification tasks.
*   It also gives us clear insight that, adding more hyperparemeter tuning to ClincalBert with large training epochs and data agumentation techinques will result it in an ideal performace.

*A confusion matrix and ROC curve are included (see below) to visualize the results of the best model.*

## Visualizations

*Confusion Matrix*
   
   ![Confusion Matrix](confusion_matrix.png)

*   *ROC Curve*
   
   ![ROC Curve](roc_curve.png)

## Code and Implementation

The project is implemented in Python and utilizes the following key libraries:

*   **pandas:** Data manipulation and analysis.
*   **scikit-learn:** Machine learning algorithms (TF-IDF, Logistic Regression, metrics).
*   **Hugging Face Transformers:** For pre-trained models (ClinicalBERT, FLAN-T5) and training utilities.
*   **PyTorch:** Deep learning framework for fine-tuning ClinicalBERT
*   **Matplotlib and Seaborn:** Data visualization.
*   **Regular Expression :** For data augmentation

The code is organized into a Jupyter Notebook (`Trail_1.ipynb`) that provides a step-by-step guide through the data preprocessing, model training, evaluation, and visualization processes.

## Future Directions

This project provides a solid foundation for future research and development in automated pathology report classification. Some potential directions include:

*   **Improved Data Augmentation:** Exploring more sophisticated data augmentation techniques to further address class imbalance.
*   **Ensemble Methods:** Combining the strengths of different models (e.g., TF-IDF + Logistic Regression and ClinicalBERT) through ensemble learning.
*   **Active Learning:** Implementing active learning strategies to selectively label the most informative data points, reducing the amount of labeled data required for high performance.
*   **Explanation and Interpretability:** Applying explainable AI (XAI) techniques to understand the factors driving the model's predictions and provide insights to medical professionals.
*   **Integration with Clinical Workflows:** Developing a user-friendly interface and API for seamless integration of the model into existing clinical workflows.

## Conclusion

This project demonstrates the potential of LLMs for automated medical text classification. The results showcase the promise of ClinicalBERT and highlight the importance of data preprocessing, feature engineering, and hyperparameter optimization. While further research is needed, this work provides a valuable stepping stone towards more efficient and accurate diagnostic processes in healthcare.

## Contact

[Rohith Matam] - [rohithmatham@gmail.com] - [https://www.linkedin.com/in/rohith-matam/] - [https://rohithmatham12.github.io/Portfolio/]

Feel free to reach out if you have any questions or suggestions!
