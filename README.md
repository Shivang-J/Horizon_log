**SQLi Attack Detector Pipeline Overview**

This project implements an SQL Injection attack detection system that processes network request data, leverages deep learning for analysis, and provides real-time notifications. The pipeline is designed for inference on unseen request logs.
The DATASET used in this project was taken from: https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks

1. Data Ingestion
- The system begins by receiving raw network request data, typically in a JSON file format. This file contains individual request records, each with details such as method (GET/POST), url, headers, and data (request body/parameters).

2. Data Parsing & Feature Extraction
- Upon ingestion, the parse_sqli_json_data function processes the raw JSON. For each request, it extracts:

    - Method: The HTTP request method (e.g., 'GET', 'POST').

    - URL: The requested Uniform Resource Locator.

    - Content: The request body or query parameters (data field from JSON).

    - Length: The character length of the content field.
    This step transforms the raw JSON into a structured Pandas DataFrame with these key features.

3. Data Preprocessing (Feature Engineering & Transformation)
- The parsed DataFrame then undergoes a series of preprocessing steps by the preprocess_inference_data function to prepare features for the deep learning model:

    - Text Embedding Generation: The content and URL fields are processed by a pre-trained DistilRoBERTa Transformer model. The [CLS] token embedding from the Transformer's output is extracted for both content and URL, providing rich, contextual numerical representations of the text. URLs are first URL-decoded to expose their true structure.

    - Method One-Hot Encoding: The categorical Method feature ('GET', 'POST', etc.) is converted into numerical one-hot encoded vectors using a pre-fitted OneHotEncoder.

    - Length Scaling: The numerical length feature is scaled using a pre-fitted StandardScaler. This standardization ensures that the length feature is on a comparable scale to other features, improving neural network training stability and performance.

    - Feature Concatenation: All processed features—scaled length, one-hot encoded Method, and the content and URL embeddings—are concatenated into a single, high-dimensional feature vector for each request.

4. Model Inference (Prediction)
- The combined feature vectors are fed as input to a pre-trained PyTorch Multi-Layer Perceptron (MLP) model. The MLP, designed for binary classification, processes these features and outputs a prediction probability (between 0 and 1) indicating the likelihood of the request being malicious (SQL Injection). A threshold (typically 0.5) is applied to these probabilities to assign a final binary label (0 for benign, 1 for malicious).

5. Output & Notification
- The final predictions (probability and label) are appended back to the original parsed DataFrame. This augmented DataFrame is then presented in a user-friendly Streamlit web application, displaying a summary of detections and detailed information for each analyzed request.

Crucially, if any malicious requests are detected, an automated email notification is triggered. This email, sent via SMTP, includes a timestamp and detailed information about each observed attack, alerting relevant stakeholders instantly.
