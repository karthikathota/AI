# Common AI Workloads

## Computer Vision

Computer vision is a key AI workload focused on enabling computers to interpret and analyze visual data. Key tasks include:

- **Image Classification**: Recognizes and categorizes objects within images.
- **Object Detection**: Identifies and locates multiple objects in visual data.
- **Face Recognition**: Identifies individuals based on facial features.
- **OCR (Optical Character Recognition)**: Extracts text from images.
- **Image Analysis**: Provides insights such as emotion detection or scene understanding.

## Natural Language Processing (NLP)

NLP enables machines to understand and process human language. Common NLP tasks include:

- **Text Classification**: Categorizes text into specific categories (e.g., sentiment analysis, spam detection).
- **Language Translation**: Automatically translates text from one language to another.
- **Speech Recognition**: Converts spoken language into text.
- **Text-to-Speech (TTS)**: Synthesizes natural-sounding speech from text input.

## Anomaly Detection

Anomaly detection identifies patterns that deviate from expected behavior. It is widely used in:

- Fraud detection
- Cybersecurity
- Health monitoring

## Predictive Analytics

Predictive analytics uses historical data to forecast future trends using:

- Time-series forecasting
- Regression analysis
- Pattern recognition

## Content Moderation

Used to detect and prevent the upload of harmful content (e.g., adult, violent, hateful). Customizable severity levels support balance with free speech.

---

# Guiding Principles of AI Development

## Fairness

AI systems should treat everyone equally and avoid disparate impacts.

## Reliability and Safety

AI must operate reliably and safely under normal and unexpected conditions.

## Privacy and Security

AI systems must ensure the privacy and security of data throughout development and deployment.

## Inclusiveness

AI should benefit all individuals, regardless of background or ability.

## Transparency

AI systems should be understandable. Users must know the system's purpose, functionality, and limitations.

## Accountability

AI developers must ensure solutions meet ethical and legal standards.

---

# AI in Azure

Azure AI Services offer prebuilt capabilities that are easy to integrate into applications. Resources include:

- **Multi-service Resource**: A bundle of AI services accessible via a single endpoint.
- **Single-service Resource**: A resource for one specific AI service (e.g., Speech, Vision).

---

# Common Machine Learning Types

## Regression

Supervised learning that predicts numeric outcomes (e.g., price, size). It identifies relationships between variables.

## Classification

Supervised learning that predicts categorical labels. Types include:

### Binary Classification

Predicts one of two outcomes (e.g., true/false). Performance is measured using:

- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Recall**: `TP / (TP + FN)`
- **Precision**: `TP / (TP + FP)`

### Multiclass Classification

Classifies observations into one of multiple categories.

## Clustering

Unsupervised learning that groups data based on similarities. Used when no labels are provided.

---

# Deep Learning

A subset of machine learning using layered neural networks to handle complex tasks. Deep learning enables:

- Image recognition
- Speech processing
- Natural language understanding

### CNNs (Convolutional Neural Networks)

CNNs use filters to extract feature maps and learn image representations. Filter weights are optimized during training.

### Multi-modal Models

Combine image and text inputs (e.g., Microsoft's Florence model). Train with large volumes of captioned images for broad applications.

---

# Azure AI Vision

Azure AI Vision offers prebuilt and customizable vision models based on the Florence foundation model. Use cases include:

- **Image Classification**
- **Object Detection**
- **Semantic Segmentation**
- **OCR (Optical Character Recognition)**

## Azure Vision Resources

1. **Azure AI Vision**: For dedicated vision use cases.
2. **Azure AI Services**: For multi-service use including vision, language, etc.

## Face Analysis

Detects and analyzes facial features. Services include:

1. **Azure AI Vision** – Face detection and bounding boxes.
2. **Azure AI Video Indexer** – Detects faces in video.
3. **Azure AI Face** – Detects, recognizes, and analyzes faces.

---

# Azure Machine Learning

A cloud-based service for building, training, and deploying ML models.

## AutoML

Automates data preprocessing, model selection, and hyperparameter tuning.

## Azure ML Designer

Drag-and-drop interface to create end-to-end ML workflows without coding.

---

# Natural Language Processing in Azure

Azure offers multiple NLP services under Azure AI Language, including:

- **Sentiment Analysis**
- **Key Phrase Extraction**
- **Named Entity Recognition (NER)**
- **Language Detection**
- **Summarization**
- **PII Detection**

## Common NLP Workloads

- **Key Phrase Extraction**: Identifies key topics.
- **Entity Recognition**: Detects objects, locations, dates in text.
- **Sentiment Analysis**: Assesses sentiment scores from 0 to 1.

## NLP Services

### Text Analytics

- Sentiment Analysis
- Key Phrase Extraction
- Language Detection
- Named Entity Recognition

### Language Understanding (LUIS)

Builds custom natural language models to interpret intents and entities.

### Translator

- Real-time translation between 70+ languages.
- Transliteration and language detection.

### Speech Services

- **Speech-to-Text**: Converts spoken language to text.
- **Text-to-Speech**: Converts text to speech.
- **Speaker Recognition**: Identifies different speakers.

## Conversational Language Understanding (CLU)

Three key concepts:

- **Utterances**: What users say.
- **Entities**: Items referred to in speech.
- **Intents**: User’s goal.

Azure Resources for CLU:

1. **Azure AI Language**: Authoring and prediction.
2. **Azure AI Services**: For predictions only.

---
