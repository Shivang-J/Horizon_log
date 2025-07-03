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

# --- SET STREAMLIT PAGE CONFIG FIRST ---
st.set_page_config(layout="wide", page_title="SQLi Attack Detector")

# --- Global Constants (Must be defined here or imported from a config file) ---
TRANSFORMER_LOCAL_PATH = './saved_distilroberta_inference'
MLP_MODEL_PATH = 'best_pytorch_mlp_model.pth'
LENGTH_SCALER_PATH = 'length_scaler2.pkl'
METHOD_ENCODER_PATH = 'method_encoder.pkl'

MAX_LENGTH = 256
BATCH_SIZE_INFERENCE = 64 # This is less relevant for single-row processing but kept for consistency

INPUT_SIZE_MLP = 1540
HIDDEN_SIZES_MLP = (512, 256, 128)
OUTPUT_SIZE_MLP = 1
DROPOUT_RATE_MLP = 0.2

# --- Model/Encoder Loading (Cached) ---
@st.cache_resource
def load_all_models_and_encoders():
    """Loads all necessary models and encoders, caching them for Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading resources on: {device}")

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

global_tokenizer, global_model, scaler_for_length, method_encoder, mlp_model, device = load_all_models_and_encoders()
print("All models and encoders loaded for Streamlit application.")

# --- Helper Function for CLS Embedding (needed by preprocess_inference_data) ---
def get_cls_embedding(text: str, tokenizer, model, device, max_length: int):
    """
    Helper function to get [CLS] embedding for a given text for inference.
    Handles empty strings by returning a zero vector of the correct dimension.
    """
    if not text:
        return torch.zeros(model.config.hidden_size).to(device).cpu().numpy()

    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)
        cls_embedding = model_output.last_hidden_state[0, 0, :]

    return cls_embedding.cpu().numpy()

# --- MODIFIED: Preprocessing Function now handles a single row ---
def preprocess_inference_data_single_row(
    df_row: pd.Series, # Expecting a single row (Series) for "real-time" processing
    tokenizer,
    model,
    scaler_for_length,
    method_encoder,
    device,
    max_length: int
) -> pd.DataFrame: # Returns a DataFrame with a single row of processed features
    """
    Preprocesses a single request (DataFrame row) for inference, consistent with training.
    Returns a DataFrame with a single row of processed features.
    """
    content_text = str(df_row['content']) if pd.notna(df_row['content']) else ''
    url_text = urllib.parse.unquote(str(df_row['URL'])) if pd.notna(df_row['URL']) else ''
    method = df_row['Method']
    length = df_row['length']

    # Get CLS embeddings for content and URL
    content_cls_embedding = get_cls_embedding(content_text, tokenizer, model, device, max_length)
    url_cls_embedding = get_cls_embedding(url_text, tokenizer, model, device, max_length)

    # Convert embeddings to DataFrames
    content_embed_df = pd.DataFrame([content_cls_embedding]).add_prefix('content_embed_')
    url_embed_df = pd.DataFrame([url_cls_embedding]).add_prefix('url_embed_')

    # One-hot encode the Method - **FIXED for warning**
    method_df_for_encoding = pd.DataFrame({'Method': [method]})
    method_encoded = method_encoder.transform(method_df_for_encoding)
    method_encoded_df = pd.DataFrame(
        method_encoded,
        columns=method_encoder.get_feature_names_out(['Method'])
    )

    # Scale the length - **FIXED for warning**
    length_df_for_scaling = pd.DataFrame({'length': [length]})
    try:
        length_scaled = scaler_for_length.transform(length_df_for_scaling)
        length_scaled_df = pd.DataFrame(length_scaled, columns=['length'])
    except Exception as e:
        # Fallback if scaler fails (e.g., zero variance during fit)
        st.warning(f"Error during length scaling for a single request: {e}. Using unscaled length for this request.")
        length_scaled_df = pd.DataFrame([[length]], columns=['length'])

    # Concatenate all features
    preprocessed_X = pd.concat([
        length_scaled_df,
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

    except json.JSONDecodeError as e:
        return pd.DataFrame(columns=['Method', 'URL', 'length', 'content'])
    except Exception as e:
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
    value=os.getenv("DEFAULT_RECIPIENT_EMAIL", "your_recipient_email@example.com"),
    key="sidebar_recipient_email_input",
    help="Enter the email address where attack alerts should be sent."
)
st.sidebar.info("Ensure EMAIL_ADDRESS and EMAIL_PASSWORD environment variables are set for sender email credentials.")

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully! Click 'Analyze Attacks' to proceed.")
    file_contents = uploaded_file.getvalue().decode("utf-8")

    if st.sidebar.button("Analyze Attacks", key="analyze_button"):
        st.header("Real-time Analysis Results")
        try:
            all_requests_df = parse_sqli_json_data(file_contents)

            if all_requests_df.empty:
                st.warning("No data found in the uploaded JSON file or parsing failed.")
                st.stop()

            st.write("--- Starting Real-time Analysis ---")

            # --- Placeholder for Real-time Metrics ---
            metrics_placeholder = st.empty()
            progress_bar = st.progress(0)
            attack_details_placeholder = st.expander("Click to view individual attack details", expanded=True)
            malicious_summary_placeholder = st.empty() # For the final attack summary table

            total_requests_processed = 0
            malicious_count = 0
            benign_count = 0
            total_accuracy_display = st.empty() # To display accuracy if we have ground truth

            all_predictions_results = [] # To store all predictions for final display
            email_attack_details_list = [] # To store details for email notification

            # Create an empty DataFrame to be updated in real-time
            real_time_df_placeholder = st.empty()
            real_time_df = pd.DataFrame(columns=['Timestamp', 'Method', 'URL', 'Content', 'Predicted_Probability', 'Predicted_Label'])


            # --- Real-time Processing Loop ---
            for index, current_request_row in all_requests_df.iterrows():
                # Simulate a delay for real-time effect - ADJUST THIS VALUE FOR SPEED
                time.sleep(0.05) # Reduced to 0.05 seconds for faster real-time updates

                total_requests_processed += 1
                current_timestamp = datetime.datetime.now()

                # Preprocess the single row
                processed_features_for_pred_single = preprocess_inference_data_single_row(
                    current_request_row, global_tokenizer, global_model, scaler_for_length, method_encoder, device, MAX_LENGTH
                )

                # Validate feature count for the single row
                if processed_features_for_pred_single.shape[1] != INPUT_SIZE_MLP:
                    st.error(f"FATAL ERROR: Mismatch in feature count after preprocessing for request {index}!")
                    st.error(f"Expected {INPUT_SIZE_MLP} features, but got {processed_features_for_pred_single.shape[1]}.")
                    st.stop() # Stop processing if a fundamental error occurs

                # Make prediction for the single row
                X_tensor_for_pred_single = torch.tensor(processed_features_for_pred_single.values, dtype=torch.float32).to(device)
                with torch.no_grad():
                    raw_output_single = mlp_model(X_tensor_for_pred_single)
                    prediction_proba_single = raw_output_single.cpu().numpy().flatten()[0]
                    predicted_label_single = (prediction_proba_single > 0.5).astype(int)

                # Update counts
                if predicted_label_single == 1:
                    malicious_count += 1
                else:
                    benign_count += 1

                # Append to the list for final DataFrame
                all_predictions_results.append({
                    'Method': current_request_row['Method'],
                    'URL': current_request_row['URL'],
                    'length': current_request_row['length'],
                    'content': current_request_row['content'],
                    'Predicted_Probability': prediction_proba_single,
                    'Predicted_Label': predicted_label_single,
                    'Timestamp': current_timestamp
                })

                # Update the real-time display DataFrame
                new_row_df = pd.DataFrame([{
                    'Timestamp': current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Method': current_request_row['Method'],
                    'URL': current_request_row['URL'],
                    'Content': current_request_row['content'],
                    'Predicted_Probability': f"{prediction_proba_single:.4f}",
                    'Predicted_Label': "Malicious 🚨" if predicted_label_single == 1 else "Benign ✅"
                }])
                real_time_df = pd.concat([new_row_df, real_time_df]).head(10) # Keep last 10 for display
                real_time_df_placeholder.dataframe(real_time_df, hide_index=True)


                # Update metrics dashboard
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Total Requests", total_requests_processed)
                    with col2: st.metric("Malicious Detections", malicious_count)
                    with col3: st.metric("Benign Requests", benign_count)

                # Update progress bar
                progress_bar.progress(min(100, int((total_requests_processed / len(all_requests_df)) * 100)))

                # Update attack details expander for malicious requests
                if predicted_label_single == 1:
                    with attack_details_placeholder:
                        st.error(f"🚨 **Potential SQLi Attack Detected!** (Request {total_requests_processed})")
                        st.write(f"- **Method:** `{current_request_row['Method']}`")
                        st.write(f"- **URL:** `{current_request_row['URL']}`")
                        st.write(f"- **Content:** `{current_request_row['content']}`")
                        st.write(f"- **Confidence:** `{prediction_proba_single:.4f}`")
                        st.write(f"- **Timestamp:** `{current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`")
                        st.markdown("---") # Separator

                    email_attack_details_list.append({
                        'method': current_request_row['Method'],
                        'url': current_request_row['URL'],
                        'content': current_request_row['content'],
                        'prob': prediction_proba_single,
                        'timestamp': current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })

            st.success(f"Analysis complete! Processed {total_requests_processed} requests.")

            # --- Final Summary and Email Trigger after all requests are processed ---
            if malicious_count > 0:
                malicious_summary_placeholder.subheader("Summary of Malicious Detections")
                malicious_df_final = pd.DataFrame(all_predictions_results)[pd.DataFrame(all_predictions_results)['Predicted_Label'] == 1]
                malicious_summary_placeholder.dataframe(malicious_df_final[['Timestamp', 'Method', 'URL', 'content', 'Predicted_Probability']])

                st.subheader("Attack Notification")
                st.warning(f"🚨 **Attack Alert!** {malicious_count} malicious requests detected.")

                if recipient_email_input and "@" in recipient_email_input:
                    with st.spinner("Sending email notification..."):
                        send_attack_notification_email(recipient_email_input, email_attack_details_list)
                else:
                    st.info("No valid recipient email provided for notifications. Email notification skipped.")
            else:
                st.info("No malicious attacks detected in this batch. 🎉")


        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON.")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            st.exception(e)

else:
    st.info("Upload a JSON file containing network requests in the sidebar and click 'Analyze Attacks' to detect SQLi.")

st.sidebar.markdown("---")
st.sidebar.info("This application analyzes network requests for potential SQL Injection attacks.")