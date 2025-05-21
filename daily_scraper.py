import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import os
import joblib
import re


DATA_DIR = 'data'
MODELS_DIR = 'models'
MASTER_SEEN_URLS_FILE = os.path.join(DATA_DIR, 'master_seen_urls.csv')
# Output file for new jobs found today, set to CSV
TODAYS_NEW_JOBS_FILE = os.path.join(DATA_DIR, 'todays_new_jobs.csv') 

VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')

# Daily check focuses on specific keywords.
KEYWORDS_TO_CHECK_DAILY = [
    "Data Scientist", "Machine Learning Engineer", "AI Engineer", "Data Analyst", "AI Researcher",
    "Data Engineer" 
]
PAGES_TO_SCRAPE_PER_KEYWORD_DAILY = 1 # Assumes only the first page is relevant for daily new jobs.

CLUSTER_MAPPING = { 
    0: "Natural Language Processing & ML",
    1: "Systems Programming (Linux/Unix) & PyTorch",
    2: "Cloud Data Services (AWS & Apache)",
    3: "DevOps & Multi-Cloud Platforms (AWS/GCP)",
    4: "Data Science & Analytics (SQL & Programming)",
    5: "Big Data Engineering (Java, Spark, Hadoop)"
}


def load_master_seen_urls(file_path):
    # Loads seen URLs from the master CSV file into a set for efficient lookup.
    seen_urls = set()
    if os.path.exists(file_path):
        try:
            df_seen = pd.read_csv(file_path)
            if 'Job_URL' in df_seen.columns:
                seen_urls.update(df_seen['Job_URL'].dropna().astype(str).tolist())
        except pd.errors.EmptyDataError: # Handles case where CSV exists but is empty
            print(f"INFO: Master seen URLs file '{file_path}' is empty. Starting with an empty set.")
        except Exception as e:
            print(f"ERROR loading master seen URLs from '{file_path}': {e}")
    else:   
        print(f"WARNING: Master seen URLs file '{file_path}' not found. Starting with an empty set.")
    return seen_urls

def append_urls_to_master_list(new_urls_list, file_path):
    if not new_urls_list:
        return
    new_urls_df = pd.DataFrame(new_urls_list, columns=['Job_URL'])
    try:
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        new_urls_df.to_csv(file_path, mode='a', header=write_header, index=False)
        print(f"Appended {len(new_urls_list)} new URLs to '{file_path}'.")
    except Exception as e:
        print(f"ERROR: Error appending URLs to '{file_path}': {e}")

def process_skills_for_prediction(raw_skills_string):
    if not isinstance(raw_skills_string, str) or not raw_skills_string.strip():
        return "" 
    skill_string_lower = raw_skills_string.lower()
    tokens = [token.strip() for token in skill_string_lower.split(',')] 
    cleaned_tokens_for_tfidf = []
    for token in tokens:
        processed_token = token.strip()
        processed_token = re.sub(r'\s*\([^)]*\)', '', processed_token).strip() 
        processed_token = re.sub(r'[^a-z0-9\s\.\+\#\-]', ' ', processed_token) 
        processed_token = ' '.join(processed_token.split()) 
        if processed_token:
            cleaned_tokens_for_tfidf.append(processed_token)
    return ' '.join(cleaned_tokens_for_tfidf)

#Daily Scraping and Categorization
def daily_karkidi_scrape_and_categorize(keyword, pages_to_try, master_seen_urls_set, tfidf_vectorizer, kmeans_model):
    # Scrapes Karkidi for a keyword, identifies new jobs, categorizes them, and returns new job details.
    # Updates master_seen_urls_set in-memory with newly found job URLs.
    headers = {'User-Agent': 'Mozilla/5.0 DailyJobScraper/1.3'} 
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    
    newly_categorized_jobs_for_keyword = []

    print(f"\nDaily check for keyword: '{keyword}'")
    page_num = 1 # We are only processing page 1 for daily checks
    url = base_url.format(page=page_num, query=keyword.replace(' ', '%20'))
    print(f"Scraping: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return newly_categorized_jobs_for_keyword # Return empty list if page fetch fails

    soup = BeautifulSoup(response.content, "html.parser")
    job_blocks = soup.find_all("div", class_="ads-details")

    if not job_blocks:
        print(f"No job blocks found for keyword '{keyword}'.")
        return newly_categorized_jobs_for_keyword

    for job_html_block in job_blocks:
        scraped_at = datetime.now().isoformat()
        title, job_url, company, location, experience, posting_date, job_type, summary, skills_raw = ["Not found"] * 9

        try:
            cmp_info_div = job_html_block.find("div", class_="cmp-info")
            if cmp_info_div:
                title_link_tag = cmp_info_div.find("a", href=lambda h: h and "job-details" in h)
                if title_link_tag:
                    h4 = title_link_tag.find("h4")
                    if h4: title = h4.get_text(strip=True)
                    url_val = title_link_tag.get('href')
                    if url_val: job_url = "https://www.karkidi.com" + url_val if url_val.startswith('/') else url_val
                
                co_tag = cmp_info_div.find("a", href=lambda h: h and "Employer-Profile" in h)
                if co_tag: company = co_tag.get_text(strip=True)
                
                map_icon = cmp_info_div.find("i", class_="fa-map-marker")
                if map_icon and map_icon.parent and map_icon.parent.name == 'p':
                    location = map_icon.parent.get_text(strip=True)
                
                exp_tag = cmp_info_div.find("p", class_="emp-exp")
                if exp_tag: experience = exp_tag.get_text(strip=True)

            hour_div = job_html_block.find("div", class_="hour-details")
            if hour_div:
                date_p = hour_div.find("p")
                if date_p: posting_date = date_p.get_text(strip=True)
                
                type_span = hour_div.find("span", class_=lambda c: c and any(k in c for k in ['fulltime','parttime','contract']))
                if type_span: job_type = type_span.get_text(strip=True)
                elif not type_span and hour_div.find("span", class_="label-warning"): 
                     type_span_alt = hour_div.find("span", class_="label-warning")
                     if type_span_alt and type_span_alt.get_text(strip=True) in ["Full Time", "Part Time"]: 
                          job_type = type_span_alt.get_text(strip=True)
            
            summary_span = job_html_block.find("span", class_="left-content", string="Summary")
            if summary_span: 
                parts = []
                curr = summary_span.find_next_sibling()
                while curr:
                    if curr.name == 'div' and 'msg-cell' in curr.get('class', []): 
                        if curr.find("span", class_="left-content", string="Key Skills"): break
                    if curr.name in ['p','ul']: parts.append(curr.get_text(separator=' ', strip=True))
                    curr = curr.find_next_sibling()
                summary = " ".join(parts).strip() if parts else (summary_span.find_next("p").get_text(strip=True) if summary_span.find_next("p") else "Not found")

            skills_span = job_html_block.find("span", class_="left-content", string="Key Skills")
            if skills_span:
                skills_p = skills_span.find_next_sibling("p", class_="text-greey")
                if skills_p: skills_raw = skills_p.get_text(strip=True)
                elif skills_span.find_next("p"): skills_raw = skills_span.find_next("p").get_text(strip=True)
                            
            if job_url != "Not found":
                if job_url not in master_seen_urls_set: 
                    master_seen_urls_set.add(job_url) 
                    
                    processed_skills = process_skills_for_prediction(skills_raw)
                    skill_vector = tfidf_vectorizer.transform([processed_skills]) 
                    cluster_label = kmeans_model.predict(skill_vector)[0]
                    category_name = CLUSTER_MAPPING.get(cluster_label, "Other/Uncategorized")

                    new_job_details = {
                        "Title": title, "Company": company, "Location": location, 
                        "Experience": experience, "Summary": summary, "Skills_Raw": skills_raw,
                        "Posting_Date": posting_date, "Job_URL": job_url, "Job_Type": job_type,
                        "Predicted_Category": category_name, 
                        "Predicted_Cluster_Label": int(cluster_label), 
                        "Daily_Scraped_Timestamp": scraped_at,
                        "Keyword_Found_Under": keyword
                    }
                    newly_categorized_jobs_for_keyword.append(new_job_details)
                    print(f"  NEW JOB (Overall): '{title}' (Category: {category_name})")
            else: 
                print(f"  Skipping job (Title: '{title}') due to 'Not found' Job_URL.")
        except Exception as e:
            print(f"  Error parsing a job block (Title: '{title if title != 'Not found' else 'Unknown'}'): {e}")
            continue 
    
    print(f"Finished keyword: '{keyword}'. Found {len(newly_categorized_jobs_for_keyword)} brand new job(s).")
    return newly_categorized_jobs_for_keyword

# Execution Logic for the Daily Scraper 
if __name__ == "__main__":
    print("Starting daily job check...")
    
    tfidf_vectorizer_loaded = None
    kmeans_model_loaded = None
    try:
        for dir_path in [MODELS_DIR, DATA_DIR]: 
            if not os.path.exists(dir_path): os.makedirs(dir_path)

        tfidf_vectorizer_loaded = joblib.load(VECTORIZER_PATH)
        kmeans_model_loaded = joblib.load(KMEANS_MODEL_PATH)
        print("INFO: Models loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load models from '{MODELS_DIR}'. Exiting. Error: {e}")
        exit() 

    master_seen_urls = load_master_seen_urls(MASTER_SEEN_URLS_FILE)
    initial_master_seen_urls_count = len(master_seen_urls)
    all_todays_new_jobs_details = [] 

    for kw in KEYWORDS_TO_CHECK_DAILY:
        new_jobs_from_kw = daily_karkidi_scrape_and_categorize(
            keyword=kw,
            pages_to_try=PAGES_TO_SCRAPE_PER_KEYWORD_DAILY, 
            master_seen_urls_set=master_seen_urls, 
            tfidf_vectorizer=tfidf_vectorizer_loaded,
            kmeans_model=kmeans_model_loaded
        )
        all_todays_new_jobs_details.extend(new_jobs_from_kw)

        if len(KEYWORDS_TO_CHECK_DAILY) > 1 and kw != KEYWORDS_TO_CHECK_DAILY[-1]: 
            print(f"Pausing for 3 seconds ")
            time.sleep(3)

    print(f"\n Daily Scraping Summary")
    if all_todays_new_jobs_details:
        print(f"Found {len(all_todays_new_jobs_details)} new job(s) today across all keywords.")
        
        # Save new jobs to a CSV file
        try:
            df_new_jobs = pd.DataFrame(all_todays_new_jobs_details)
            df_new_jobs.to_csv(TODAYS_NEW_JOBS_FILE, index=False)
            print(f"Today's new jobs saved to '{TODAYS_NEW_JOBS_FILE}'")
        except Exception as e:
            print(f"ERROR: Error saving today's new jobs to CSV: {e}")

        newly_added_urls_to_master_list = [job['Job_URL'] for job in all_todays_new_jobs_details if job['Job_URL'] != "Not found"]
            
        if newly_added_urls_to_master_list:
            append_urls_to_master_list(newly_added_urls_to_master_list, MASTER_SEEN_URLS_FILE)
        else:
            print("No new valid URLs to append to the master list.")
    else:
        print("No new jobs found today across all keywords.")

    print(f"Master seen URLs count before run (from file): {initial_master_seen_urls_count}")
    print(f"Master seen URLs count after run (including new ones): {len(master_seen_urls)}") 
    print("Daily job check finished.")