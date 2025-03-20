import pandas as pd
import requests
from bs4 import BeautifulSoup
import os




api_key = ['0ndxcfLK5ncYlIqoibGzK8QmjgFpjK0zwo4dSFFA', 'mMecbd2llzpRG17qGE2QwbSZwPGoCaM06aEqeITx', 'h310atobblXmPgMqe48lYXYh14cDWc7wSN08VIHE', 'FeRN4xyXbhbkvKvbLbqJhm6uBoyo7a1QJuSwIibA', 'ufICLByVrzRqz1y9SrFLI6IAGhF9KEf1v7cckfvB']



# Function to get titles of the bill
def get_bill_titles(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/titles?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        titles_data = response.json()
        titles = [title['title'] for title in titles_data.get('titles', [])]
        return titles
    return []

# Function to get summaries of the bill
def get_bill_summaries(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/summaries?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        summaries_data = response.json()
        if summaries_data.get('summaries'):
            return summaries_data['summaries'][0]['text']
    return ""

# Function to get committees of the bill
def get_bill_committees(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/committees?api_key={api_key[0]}'
    response = requests.get(url)
    if response.status_code == 200:
        committees_data = response.json()
        committees = []
        for committee in committees_data.get('committees', []):
            committees.append(tuple([
                committee.get('chamber'),
                committee.get('name')
            ]))
        return committees
    return []

# Function to get cosponsors of the bill
def get_bill_cosponsors(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/cosponsors?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}?api_key={api_key[iteration % len(api_key)]}'
    response2 = requests.get(url)
    cosponsors = []
    if response2.status_code == 200:
        bill_data = response2.json().get('bill', {})
        for sponsor in bill_data.get('sponsors', []):
            cosponsors.append(tuple([
                sponsor.get('bioguideId', ''),
                sponsor.get('fullName', ''),
                sponsor.get('party', ''),
                sponsor.get('state', ''),
                sponsor.get('firstName', ''),
                sponsor.get('lastName', '')
            ]))
    if response.status_code == 200:
        cosponsors_data = response.json()
        for cosponsor in cosponsors_data.get('cosponsors', []):
            cosponsors.append(tuple([
                cosponsor.get('bioguideId', ''),
                cosponsor.get('fullName', ''),
                cosponsor.get('party', ''),
                cosponsor.get('state', ''),
                cosponsor.get('firstName', ''),
                cosponsor.get('lastName', '')
            ]))
        return cosponsors
    return []

# Function to get subjects of the bill
def get_bill_subjects(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/subjects?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        subjects_data = response.json()
        subjects = [subject['name'] for subject in subjects_data.get('subjects', {}).get('legislativeSubjects', [])]
        policy_area = subjects_data.get('subjects', {}).get('policyArea', {}).get('name')
        if policy_area:
            subjects.append(tuple(policy_area))
        return subjects
    return []

# Function to get text of the bill
def get_bill_text(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/text?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        text_data = response.json()
        if text_data.get('textVersions'):
            earliest_version = text_data['textVersions'][-1]  
            for format in earliest_version['formats']:
                if format['type'] == 'Formatted Text':
                    text_url = format['url']
                    text_response = requests.get(text_url)
                    if text_response.status_code == 200:
                        soup = BeautifulSoup(text_response.content, 'html.parser')
                        return soup.get_text()
    return ""




def extract_unique_bill_info(input_dir, output_file, old_csv = None, old_csv_with_api = None):
    if old_csv is not None:
        unique_records = set(pd.read_csv(old_csv).apply(tuple, axis=1))
        unique_records_with_api = set(pd.read_csv(old_csv_with_api).apply(tuple, axis=1))
    else: 
        unique_records = set()
        unique_records_with_api = set()
    iteration = 0
    congressman_number = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            try:
                df = pd.read_csv(file_path)
            except:
                congressman_number = congressman_number+1
                print(f"CSV READ ERROR ON CONGRESSMAN NUMBER {congressman_number}, {filename}")
                continue
            congressman_number = congressman_number+1

            for index, row in df.iterrows():
                try:
                    congress = row['Congress']
                    bill_type = row['Bill Type']
                    bill_number = int(row['Bill Number'])
                except:
                    continue
                if not (bill_number > 0):
                    continue
                if (congress, bill_type, bill_number) in unique_records:
                    print(f"Congressman {congressman_number}, {filename}, Duplicate found and ignored: {(congress, bill_type, bill_number)}")
                    continue
                unique_records.add((congress, bill_type, bill_number))
                try:
                    titles = tuple(get_bill_titles(congress, bill_type, bill_number, iteration))
                    summaries = get_bill_summaries(congress, bill_type, bill_number, iteration)
                    committees = tuple(get_bill_committees(congress, bill_type, bill_number, iteration))
                    cosponsors = tuple(get_bill_cosponsors(congress, bill_type, bill_number, iteration))
                    subjects = tuple(get_bill_subjects(congress, bill_type, bill_number, iteration))
                    text = get_bill_text(congress, bill_type, bill_number, iteration)
                except:
                    unique_records.remove((congress, bill_type, bill_number))
                    print(f"Congressman {congressman_number}, {filename}, API Error: {(congress, bill_type, bill_number)}")
                    continue
                unique_records_with_api.add((congress, bill_type, bill_number, titles, summaries, committees, cosponsors, subjects, text))
                iteration = iteration + 1
                
                if iteration % 10 == 0:
                    unique_df = pd.DataFrame(list(unique_records_with_api), columns=['congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'committees', 'cosponsors', 'subjects', 'text'])
                    unique_df.to_csv(output_file, index=False)
                    restart_df = pd.DataFrame(list(unique_records), columns=['congress', 'bill_type', 'bill_number'])
                    restart_df.to_csv('restart.csv', index=False)
                print(f"Congressman {congressman_number}, {filename}, {iteration}. Added {(congress, bill_type, bill_number)}, {iteration*8} requests used, key used: {api_key[(iteration-1) % len(api_key)]}")

    unique_df = pd.DataFrame(list(unique_records_with_api), columns=['congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'committees', 'cosponsors', 'subjects', 'text'])
    unique_df.to_csv(output_file, index=False)
    restart_df = pd.DataFrame(list(unique_records), columns=['congress', 'bill_type', 'bill_number'])
    restart_df.to_csv('restart.csv', index=False)
    return len(unique_df)
 

"""
input_directory = 'data_collection/congressman_data'
output_csv = 'data_collection/billinfo.csv'
old = "restart.csv"
old_api = "data_collection/billinfo.csv"
print(f"Found {extract_unique_bill_info(input_directory, output_csv, old, old_api)} unique bills")
"""