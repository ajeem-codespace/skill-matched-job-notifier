import time
import pandas as pd
import json
import os
import smtplib 
from email.mime.text import MIMEText 
from datetime import datetime

DATA_DIR = 'data'
TODAYS_NEW_JOBS_FILE = os.path.join(DATA_DIR, 'todays_new_jobs.csv') 
USER_PREFERENCES_FILE = os.path.join(DATA_DIR, 'user_preferences.json')

# Email configuration 
SENDER_EMAIL = os.environ.get('MAIL_USERNAME')
SENDER_APP_PASSWORD = os.environ.get('MAIL_PASSWORD') 
SMTP_SERVER = 'smtp.gmail.com' 
SMTP_PORT = 587 


def load_new_jobs(file_path):
    """Loads today's newly found jobs from the CSV file."""
    if not os.path.exists(file_path):
        print(f"INFO: '{file_path}' not found. No new jobs to process for notifications.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        # Ensure 'Predicted_Category' and other essential columns exist
        if 'Predicted_Category' not in df.columns or 'Title' not in df.columns or 'Job_URL' not in df.columns:
            print(f"ERROR: '{file_path}' is missing essential columns (e.g., Title, Job_URL, Predicted_Category).")
            return pd.DataFrame()
        print(f"Loaded {len(df)} new jobs from '{file_path}'.")
        return df
    except pd.errors.EmptyDataError:
        print(f"INFO: '{file_path}' is empty. No new jobs to process.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Could not load new jobs from '{file_path}': {e}")
        return pd.DataFrame()

def load_user_preferences(file_path):
    #Loads user preferences from the JSON file.
    if not os.path.exists(file_path):
        print(f"ERROR: User preferences file '{file_path}' not found.")
        return []
    try:
        with open(file_path, 'r') as f:
            preferences = json.load(f)
        print(f"Loaded preferences for {len(preferences)} user(s) from '{file_path}'.")
        return preferences
    except Exception as e:
        print(f"ERROR: Could not load user preferences from '{file_path}': {e}")
        return []

def send_email_notification(recipient_email, user_name, job_title, job_url, job_category, company):
    #Sends an email notification for a job match.
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
        print("ERROR: Sender email or password not configured. Cannot send email.")
        return False

    subject = f"New Job Alert: '{job_title}' in {job_category}"
    body = f"""
    Hello {user_name if user_name else 'User'},

    A new job matching your preferred category '{job_category}' has been posted:

    Title: {job_title}
    Company: {company if company and company != 'Not found' else 'N/A'}
    Category: {job_category}
    Link: {job_url}

    Consider applying if it interests you!

    Best regards,
    Your Skill-Matched Job Notifier
    (Sent on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls() # Secure the connection
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print(f"Successfully sent email to {recipient_email} for job: '{job_title}'")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send email to {recipient_email}: {e}")
        return False

# Main Notification Logic 
if __name__ == "__main__":
    print("Starting Notification Sender...")

    new_jobs_df = load_new_jobs(TODAYS_NEW_JOBS_FILE)
    user_preferences = load_user_preferences(USER_PREFERENCES_FILE)

    if new_jobs_df.empty or not user_preferences:
        print("No new jobs found or no user preferences defined. Exiting notification process.")
        exit()
    
    notifications_sent_count = 0
    for user_pref in user_preferences:
        user_email = user_pref.get("email")
        user_name = user_pref.get("name", "User") # Default to "User" if name is missing
        preferred_categories = user_pref.get("preferred_categories", [])

        if not user_email or not preferred_categories:
            print(f"Skipping user due to missing email or preferred categories: {user_pref}")
            continue

        print(f"\nChecking jobs for user: {user_email} (Name: {user_name}) with preferences: {preferred_categories}")
        
        for index, job in new_jobs_df.iterrows():
            job_title = job.get('Title', 'N/A')
            job_url = job.get('Job_URL', '#')
            job_category = job.get('Predicted_Category', 'N/A')
            job_company = job.get('Company', 'N/A') 

            if job_category in preferred_categories:
                print(f"  MATCH FOUND: Job '{job_title}' in category '{job_category}' matches preferences of {user_email}.")

                if send_email_notification(user_email, user_name, job_title, job_url, job_category, job_company):
                    notifications_sent_count += 1
                time.sleep(1) # Small delay between sending emails if multiple matches for a user

    print(f"\nNotification process finished. Total notifications sent: {notifications_sent_count}")