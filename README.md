**SQLi Attack Detector Pipeline Overview**

This project implements an SQL Injection attack detection system that processes network request data, leverages deep learning for analysis, and provides real-time notifications. The pipeline is designed for inference on unseen request logs.

The DATASET used in this project was taken from: https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks. After parsing that data, a file named "Final_Data" was created, on which the whole model is trained.

It contains columns: **Method	length	content	URL	Label**


**SETUP INSTRUCTIONS**

Follow these steps to get the SQLi Attack Detector up and running on your local machine.
Prerequisites

    Python 3.8 or higher installed.
    - All required Python libraries listed in requirements.txt (will be installed automatically).

    pip (Python package installer).

    Crucially, you need your trained model and preprocessor files:

        A directory named saved_distilroberta_inference/ containing your saved DistilRoBERTa model and tokenizer.

        best_pytorch_mlp_model.pth (your trained PyTorch MLP model weights).

        length_scaler2.pkl (your saved StandardScaler object).

        method_encoder.pkl (your saved OneHotEncoder object).

        These files MUST be placed in the same directory as your app.py file.

1. Clone the Repository (or create project directory)

If this is a new project, create a directory:

    mkdir sqli_detector
    cd sqli_detector

Then, place your app.py file and all the prerequisite model/preprocessor files (saved_distilroberta_inference/, best_pytorch_mlp_model.pth, length_scaler2.pkl, method_encoder.pkl) into this sqli_detector directory.

2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies:

    python -m venv venv

3. Activate the Virtual Environment

    On Windows:

        .\venv\Scripts\activate

    On macOS/Linux:

        source venv/bin/activate

4. Install Dependencies

With your virtual environment activated, install all required Python packages listed in 'requirements.txt':

    pip install -r requirements.txt

(Note: torch installation might vary based on your CUDA/CPU setup. Refer to PyTorch's official website for specific instructions if you encounter issues.)
5. Configure Email Notifications (Optional but Recommended)

For the email notification feature to work, you need to set up environment variables for your sender email credentials.

    Using Gmail (Recommended for ease):

        Go to your Google Account.

        Navigate to "Security".

        Under "How you sign in to Google", ensure "2-Step Verification" is ON.

        Once 2-Step Verification is on, an "App passwords" option will appear (you might need to search for it or refresh the page). Click on it.

        Generate a new app password. This is a 16-character code.

        Set the following environment variables in your terminal before running the Streamlit app:

            On Windows (Command Prompt/PowerShell):

            set EMAIL_ADDRESS=your_sender_email@gmail.com
            set EMAIL_PASSWORD=your_generated_app_password

            (For PowerShell, you might use $env:EMAIL_ADDRESS="your_sender_email@gmail.com")

            On macOS/Linux (Bash/Zsh):

            export EMAIL_ADDRESS="your_sender_email@gmail.com"
            export EMAIL_PASSWORD="your_generated_app_password"

        (Replace your_sender_email@gmail.com and your_generated_app_password with your actual details.)

        Note: These variables are session-specific. If you close your terminal, you'll need to set them again or add them to your shell's profile (.bashrc, .zshrc, etc.) for persistence.

How to Run the Application

Once all prerequisites are met and dependencies are installed:

    Navigate to your project directory in the terminal (where app.py is located).

    Ensure your virtual environment is activated.

    Run the Streamlit application:

    streamlit run app.py

    This command will open the application in your default web browser.

**HOW THE MODEL WORKS** 
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
