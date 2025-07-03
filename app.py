import streamlit as st # Keep this import at the very top for clarity
import pandas as pd
import json
import io
import datetime
import time

# --- Essential Imports ---
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns # Added for better statistical plots

# --- SET STREAMLIT PAGE CONFIG FIRST ---
st.set_page_config(layout="wide", page_title="SQLi Attack Detector")

# --- Global Constants (Must be defined here or imported from a config file) ---
# Path where your DistilRoBERTa model and tokenizer are saved locally
TRANSFORMER_LOCAL_PATH = './saved_distilroberta_inference'
# Path where your trained MLP weights are saved
MLP_MODEL_PATH = 'best_pytorch_mlp_model.pth'
# Path to your saved StandardScaler for 'length'
LENGTH_SCALER_PATH = 'length_scaler2.pkl'
# Path to your saved OneHotEncoder for 'Method'
METHOD_ENCODER_PATH = 'method_encoder.pkl'

# Transformer parameters (must match training)
MAX_LENGTH = 256
BATCH_SIZE_INFERENCE = 64 # This is crucial for batch processing

# MLP Model Architecture parameters (must match training)
INPUT_SIZE_MLP = 1540 # Total features: 1 (length) + 2 (one-hot method) + 768 (content CLS) + 768 (URL CLS)
HIDDEN_SIZES_MLP = (512, 256, 128)
OUTPUT_SIZE_MLP = 1 # Binary classification
DROPOUT_RATE_MLP = 0.2

# --- Model/Encoder Loading (Cached) ---
@st.cache_resource
def load_all_models_and_encoders():
    """Loads all necessary models and encoders, caching them for Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading resources on: {device}") # This will print to your terminal where streamlit is running

    # Initialize all variables to None to prevent UnboundLocalError
    tokenizer, model = None, None
    scaler_for_length, method_encoder = None, None
    mlp_model = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_LOCAL_PATH)
        model = AutoModel.from_pretrained(TRANSFORMER_LOCAL_PATH)
        model.eval()
        model.to(device)
        print("DistilRoBERTa loaded.")
    except Exception as e:
        st.error(f"Error loading Transformer models from '{TRANSFORMER_LOCAL_PATH}': {e}. "
                 "Please ensure the path is correct and models are saved.")
        st.stop()

    try:
        scaler_for_length = joblib.load(LENGTH_SCALER_PATH)
        method_encoder = joblib.load(METHOD_ENCODER_PATH)
        print("Preprocessing encoders loaded.")
    except FileNotFoundError as e:
        st.error(f"Error loading preprocessor files: {e}. "
                 f"Ensure '{LENGTH_SCALER_PATH}' and '{METHOD_ENCODER_PATH}' exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading preprocessing tools: {e}")
        st.stop()

    # Re-define SimpleMLP class (must be identical to training and placed before instantiation)
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
            super(SimpleMLP, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
        def forward(self, x):
            return self.network(x)

    try:
        mlp_model = SimpleMLP(INPUT_SIZE_MLP, HIDDEN_SIZES_MLP, OUTPUT_SIZE_MLP, DROPOUT_RATE_MLP).to(device)
        # Add weights_only=True as recommended by PyTorch for security
        mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True))
        mlp_model.eval()
        print("MLP model loaded.")
    except FileNotFoundError as e:
        st.error(f"MLP model file not found: {e}. Ensure '{MLP_MODEL_PATH}' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading MLP model: {e}")
        st.stop()

    return tokenizer, model, scaler_for_length, method_encoder, mlp_model, device

# Load all resources at startup
global_tokenizer, global_model, scaler_for_length, method_encoder, mlp_model, device = load_all_models_and_encoders()
print("All models and encoders loaded for Streamlit application.")

# --- Preprocessing Function (for batch processing, as it was previously) ---
def preprocess_inference_data(
    df: pd.DataFrame,
    tokenizer,
    model,
    scaler_for_length,
    method_encoder,
    device,
    batch_size: int,
    max_length: int
) -> pd.DataFrame:
    """
    Preprocesses new inference data in a DataFrame, consistent with training.
    This version processes content and URLs in batches for efficiency.
    """
    content_texts = df['content'].fillna('').astype(str).tolist()
    url_texts = [urllib.parse.unquote(str(url)) for url in df['URL'].fillna('')]

    all_content_cls_embeddings = []
    all_url_cls_embeddings = []

    num_samples = len(df)
    for i in range(0, num_samples, batch_size):
        batch_content = content_texts[i:i + batch_size]
        batch_urls = url_texts[i:i + batch_size]

        with torch.no_grad():
            # Process Content Batch
            encoded_content_batch = tokenizer(
                batch_content, return_tensors='pt', truncation=True,
                padding='longest', max_length=max_length
            ).to(device)
            content_output = model(**encoded_content_batch)
            batch_content_cls_embeds = content_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_content_cls_embeddings.extend(batch_content_cls_embeds)
            del content_output, encoded_content_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process URL Batch
            encoded_url_batch = tokenizer(
                batch_urls, return_tensors='pt', truncation=True,
                padding='longest', max_length=max_length
            ).to(device)
            url_output = model(**encoded_url_batch)
            batch_url_cls_embeds = url_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_url_cls_embeddings.extend(batch_url_cls_embeds)
            del url_output, encoded_url_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    content_embed_df = pd.DataFrame(all_content_cls_embeddings, index=df.index).add_prefix('content_embed_')
    url_embed_df = pd.DataFrame(all_url_cls_embeddings, index=df.index).add_prefix('url_embed_')

    method_encoded = method_encoder.transform(df[['Method']])
    method_encoded_df = pd.DataFrame(
        method_encoded,
        columns=method_encoder.get_feature_names_out(['Method']),
        index=df.index
    )

    df_temp_for_scaling = df.copy()

    try:
        # Reshape to 2D array if scaler expects it (StandardScaler expects 2D)
        df_temp_for_scaling['length_scaled'] = scaler_for_length.transform(df_temp_for_scaling[['length']])
    except Exception as e:
        st.warning(f"Error during length scaling: {e}. This likely means your scaler was fitted on zero variance data. Using unscaled length.")
        # Fallback to unscaled length, filling NaN with 0 if any
        df_temp_for_scaling['length_scaled'] = df_temp_for_scaling['length'].fillna(0)


    # Concatenate all features
    preprocessed_X = pd.concat([
        df_temp_for_scaling[['length_scaled']].rename(columns={'length_scaled': 'length'}), # Rename back to 'length' if needed, or adjust MLP
        method_encoded_df,
        content_embed_df,
        url_embed_df
    ], axis=1)

    return preprocessed_X

# --- JSON Parser Function (as previously defined) ---
def parse_sqli_json_data(json_data_input: str) -> pd.DataFrame:
    """
    Parses raw JSON string containing network requests into a Pandas DataFrame
    with 'Method', 'URL', 'length', and 'content' columns.
    """
    parsed_records = []
    try:
        requests_data = json.loads(json_data_input)

        if not isinstance(requests_data, list):
            if isinstance(requests_data, dict):
                requests_data = [requests_data]
            else:
                return pd.DataFrame(columns=['Method', 'URL', 'length', 'content'])

        for req in requests_data:
            method = req.get('method', 'UNKNOWN').upper()
            url = req.get('url', '')
            content = req.get('data', '')

            if not isinstance(content, str):
                content_str = json.dumps(content) if content is not None else ''
            else:
                content_str = content

            length = len(content_str)

            parsed_records.append({
                'Method': method,
                'URL': url,
                'length': length,
                'content': content_str
            })

        return pd.DataFrame(parsed_records)

    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format. Please upload a valid JSON file.")
        return pd.DataFrame(columns=['Method', 'URL', 'length', 'content'])
    except Exception as e:
        st.error(f"Error parsing JSON data: {e}")
        return pd.DataFrame(columns=['Method', 'URL', 'length', 'content'])

# --- Email Notification Function ---
def send_attack_notification_email(recipient_email: str, attack_details: list):
    """
    Sends an email notification about detected SQLi attacks.
    """
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        st.error("Email sending skipped: Sender email or password not set in environment variables (EMAIL_ADDRESS, EMAIL_PASSWORD).")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 SQLi Attack Alert! - {len(attack_details)} Malicious Requests Detected"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    html_content = """
    <html>
        <body>
            <h3>SQLi Attack Detection Alert!</h3>
            <p>The SQLi Attack Detector observed suspicious activity:</p>
            <ul>
    """
    for detail in attack_details:
        html_content += f"""
                <li>
                    <strong>Timestamp:</strong> {detail.get('timestamp', 'N/A')}<br>
                    <strong>Method:</strong> {detail.get('method', 'N/A')}<br>
                    <strong>URL:</strong> <code>{detail.get('url', 'N/A')}</code><br>
                    <strong>Content (if POST):</strong> <code>{detail.get('content', 'N/A')}</code><br>
                    <strong>Prediction Confidence:</strong> {detail.get('prob', 'N/A'):.4f}
                </li><br>
        """
    html_content += """
            </ul>
            <p>Please investigate these requests.</p>
            <p><i>This is an automated notification from your SQLi Detector.</i></p>
        </body>
    </html>
    """

    part1 = MIMEText(html_content, "html")
    msg.attach(part1)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        st.success(f"Email notification sent to {recipient_email}!")
    except Exception as e:
        st.error(f"Failed to send email notification: {e}")
        st.info("Check your email sender details, app password (if applicable), and SMTP server settings.")


# --- Main Streamlit App Layout ---
st.title("SQLi Attack Detector 🛡️")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your JSON file containing requests", type=["json"])

# --- Email Recipient Input in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Email Notifications")
recipient_email_input = st.sidebar.text_input(
    "Recipient Email for Alerts:",
    value=os.getenv("DEFAULT_RECIPIENT_EMAIL", "your_recipient_email@example.com"), # Default from env or placeholder
    key="sidebar_recipient_email_input",
    help="Enter the email address where attack alerts should be sent."
)
st.sidebar.info("Ensure EMAIL_ADDRESS and EMAIL_PASSWORD environment variables are set for sender email credentials.")

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully! Click 'Analyze Attacks' to proceed.")
    file_contents = uploaded_file.getvalue().decode("utf-8")

    if st.sidebar.button("Analyze Attacks", key="analyze_button"):
        st.header("Analysis Results")
        try:
            # Parse the JSON data
            new_data_df = parse_sqli_json_data(file_contents)

            if new_data_df.empty:
                st.warning("No data found in the uploaded JSON file or parsing failed.")
                st.stop()

            st.write("--- Original Parsed Data Head ---")
            st.dataframe(new_data_df.head())

            # Preprocess the data, passing the loaded models/encoders
            processed_features_for_pred = preprocess_inference_data(
                new_data_df, global_tokenizer, global_model, scaler_for_length, method_encoder, device,
                BATCH_SIZE_INFERENCE, MAX_LENGTH
            )

            # Validate the shape of preprocessed features before passing to MLP
            if processed_features_for_pred.shape[1] != INPUT_SIZE_MLP:
                st.error(f"FATAL ERROR: Mismatch in feature count after preprocessing!")
                st.error(f"Expected {INPUT_SIZE_MLP} features, but got {processed_features_for_pred.shape[1]}.")
                st.error("Please ensure your preprocessing steps (esp. one-hot encoding categories and embeddings) are consistent with training.")
                st.stop()

            # Make predictions
            X_tensor_for_pred = torch.tensor(processed_features_for_pred.values, dtype=torch.float32).to(device)
            mlp_model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                raw_outputs = mlp_model(X_tensor_for_pred)
                predictions_proba = raw_outputs.cpu().numpy().flatten()
                predicted_labels = (predictions_proba > 0.5).astype(int)

            # Add predictions and timestamp to the original DataFrame
            predictions_df = new_data_df.copy()
            predictions_df['Predicted_Probability'] = predictions_proba
            predictions_df['Predicted_Label'] = predicted_labels
            # Convert numeric labels to human-readable text for display
            predictions_df['Predicted_Label_Text'] = predictions_df['Predicted_Label'].map({1: 'Malicious 🚨', 0: 'Benign ✅'})
            predictions_df['Timestamp'] = datetime.datetime.now() # Add timestamp for notification

            # --- Summary and Visualizations ---
            st.subheader("Summary of Detections")
            total_requests = len(predictions_df)
            malicious_requests = predictions_df[predictions_df['Predicted_Label'] == 1]
            num_malicious = len(malicious_requests)
            num_benign = total_requests - num_malicious

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Requests Analyzed", total_requests)
            with col2:
                st.metric("Malicious Requests Detected", num_malicious)
            with col3:
                st.metric("Benign Requests", num_benign)

            st.markdown("---")

            st.subheader("Classification Overview (Counts)")
            if total_requests > 0:
                classification_counts = predictions_df['Predicted_Label_Text'].value_counts()
                fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
                # Ensure the order is consistent, e.g., Malicious then Benign if both exist
                order = ['Malicious 🚨', 'Benign ✅']
                # Filter counts to only include existing labels and maintain order
                plot_data = classification_counts.reindex(order).dropna()

                sns.barplot(x=plot_data.index, y=plot_data.values,
                            palette={'Malicious 🚨': '#FF6347', 'Benign ✅': '#4682B4'},
                            ax=ax_bar)
                ax_bar.set_title("Total Request Classifications", fontsize=14, weight='bold')
                ax_bar.set_xlabel("Classification", fontsize=12)
                ax_bar.set_ylabel("Number of Requests", fontsize=12)
                # Add counts on top of bars
                for p in ax_bar.patches:
                    ax_bar.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                st.pyplot(fig_bar)
                plt.close(fig_bar)


            st.markdown("---")

            # Histogram for Prediction Confidence - Now with potential for different colors
            st.subheader("Prediction Confidence Distribution")
            if not predictions_df.empty:
                fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
                sns.histplot(data=predictions_df, x='Predicted_Probability', hue='Predicted_Label_Text',
                             bins=20, kde=True, ax=ax_hist, palette={'Malicious 🚨': '#FF6347', 'Benign ✅': '#4682B4'})
                ax_hist.set_title('Distribution of Prediction Probabilities by Class', fontsize=14, weight='bold')
                ax_hist.set_xlabel('Predicted Probability', fontsize=12)
                ax_hist.set_ylabel('Number of Requests', fontsize=12)
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            st.markdown("---")

            st.subheader("All Analyzed Requests")

            if not predictions_df.empty:
                # Display the full DataFrame without filters
                st.dataframe(predictions_df[['Timestamp', 'Method', 'URL', 'content', 'Predicted_Probability', 'Predicted_Label_Text']],
                                 use_container_width=True, hide_index=True)

                # Display top malicious requests if any are present
                if num_malicious > 0: # Use num_malicious from summary section
                    st.subheader("Top 10 Malicious Requests by Confidence")
                    top_malicious = predictions_df[predictions_df['Predicted_Label'] == 1].sort_values(
                        by='Predicted_Probability', ascending=False
                    ).head(10)
                    st.dataframe(top_malicious[['Timestamp', 'Method', 'URL', 'content', 'Predicted_Probability']],
                                     use_container_width=True, hide_index=True)
                else:
                    st.info("No malicious requests were detected in this batch to display a 'Top 10'.")
            else:
                st.info("No requests were processed for detailed display.")

            # --- Specific Bar Chart for Malicious Methods ---
            if num_malicious > 0:
                st.markdown("---")
                st.subheader("HTTP Methods of Detected Malicious Requests")
                method_counts = malicious_requests['Method'].value_counts()
                fig_mal_method, ax_mal_method = plt.subplots(figsize=(7, 5))
                sns.barplot(x=method_counts.index, y=method_counts.values, palette='viridis', ax=ax_mal_method)
                ax_mal_method.set_title("Methods Used in Malicious Requests", fontsize=14, weight='bold')
                ax_mal_method.set_xlabel("HTTP Method", fontsize=12)
                ax_mal_method.set_ylabel("Count", fontsize=12)
                for p in ax_mal_method.patches:
                    ax_mal_method.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                           ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                st.pyplot(fig_mal_method)
                plt.close(fig_mal_method)


            # --- Notification and Email Trigger ---
            if num_malicious > 0:
                st.subheader("Email Notification")
                st.warning(f"🚨 **Attack Alert!** {num_malicious} malicious requests observed.")

                email_attack_details = []
                # Prepare details for email, focusing on malicious ones
                for idx, row in malicious_requests.iterrows():
                    email_attack_details.append({
                        'method': row['Method'],
                        'url': row['URL'],
                        'content': row['content'],
                        'prob': row['Predicted_Probability'],
                        'timestamp': row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    })

                # Automatically trigger email if attacks are detected AND a valid recipient is provided
                if recipient_email_input and "@" in recipient_email_input:
                    with st.spinner("Sending email notification..."):
                        send_attack_notification_email(recipient_email_input, email_attack_details)
                else:
                    st.info("No valid recipient email provided for notifications. Email notification skipped.")

            else:
                st.info("No malicious attacks detected in this batch. 🎉")

        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.exception(e)

else: # Only show this initial message when no file is uploaded yet
    st.info("Upload a JSON file containing network requests in the sidebar and click 'Analyze Attacks' to detect SQLi.")

st.sidebar.markdown("---")
st.sidebar.info("This application analyzes network requests for potential SQL Injection attacks.")