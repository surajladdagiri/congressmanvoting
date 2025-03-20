import os
import json
import pandas as pd
import sys

def extract_votes(voter_id_hr, voter_id_s, first_name, last_name, state, party, data_dir='data_collection/votes_data'):
    records = []
    
    for congress in range(102, 120):
        congress_dir = os.path.join(data_dir, str(congress), 'votes')
        if not os.path.exists(congress_dir):
            continue
        for year in os.listdir(congress_dir):
            if year.startswith('.'):
                continue
            
            year_dir = os.path.join(congress_dir, year)
            for vote_folder in os.listdir(year_dir):
                if vote_folder.startswith('.'):
                    continue
                if vote_folder.startswith('s') and voter_id_s != None:
                    vote_dir = os.path.join(year_dir, vote_folder)
                    json_file = os.path.join(vote_dir, 'data.json')
                    if not os.path.exists(json_file):
                        print(json_file)
                        sys.exit("JSON file not found")
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if data.get('category') != 'passage' and data.get('category') != 'passage-suspension' and data.get('category') != 'veto-override':
                            continue
                        category = data.get('category')
                        bill = data.get('bill', {})
                        bill_type = bill.get('type')
                        bill_number = bill.get('number')
                        member_vote = None
                        for vote_type in ['Yea', 'Nay', 'Not Voting', 'Present']:
                            try:
                                if voter_id_s in [voter['id'] for voter in data['votes'].get(vote_type, [])]:
                                    member_vote = vote_type
                                    break
                            except:
                                continue
                        if member_vote:
                            record = {
                                'Category:': category,
                                'Congress': congress,
                                'Bill Type': bill_type,
                                'Bill Number': bill_number,
                                'Vote Position': member_vote
                            }
                            records.append(record)
                            print(f"Parsing through Congressman {first_name} {last_name} ({party}, {state}) Congress {congress} vote number {data.get('chamber', '')}{data.get('number', '')}, position is {member_vote}" )
                else:
                    vote_dir = os.path.join(year_dir, vote_folder)
                    json_file = os.path.join(vote_dir, 'data.json')
                    if not os.path.exists(json_file):
                        print(json_file)
                        sys.exit("JSON file not found")
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if data.get('category') != 'passage' and data.get('category') != 'passage-suspension' and data.get('category') != 'veto-override':
                            continue
                        category = data.get('category')
                        bill = data.get('bill', {})
                        bill_type = bill.get('type')
                        bill_number = bill.get('number')
                        member_vote = None
                        for vote_type in ['Yea', 'Nay', 'Not Voting', 'Present']:
                            try:
                                if voter_id_hr in [voter['id'] for voter in data['votes'].get(vote_type, [])]:
                                    member_vote = vote_type
                                    break
                            except:
                                continue
                        if member_vote:
                            record = {
                                'Category:': category,
                                'Congress': congress,
                                'Bill Type': bill_type,
                                'Bill Number': bill_number,
                                'Vote Position': member_vote
                            }
                            records.append(record)
                            print(f"Parsing through Congressman {first_name} {last_name} ({party}, {state}) Congress {congress} vote number {data.get('chamber', '')}{data.get('number', '')}, position is {member_vote}" )
    if len(records) == 0:
        return
    df = pd.DataFrame(records)
    df.to_csv(os.path.join('data_collection/congressman_data', f"{first_name}_{last_name}_{state}_{party}_{legislator_type}_{voter_id_hr}_{voter_id_s}.csv"), index=False)
                    
        


start_congress = 102
input_csv = 'data_collection/legislators-current.csv'
legislators_df = pd.read_csv(input_csv)
for index, row in legislators_df.iterrows():
    first_name = row['first_name']
    last_name = row['last_name']
    state = row['state']
    party = row['party']
    legislator_type = row['type']
    lis_id = row['lis_id']
    bioguide_id = row['bioguide_id']
    if legislator_type == "sen":
        chamber = 's'
    else:
        chamber = 'h'
        lis_id = None
    extract_votes(bioguide_id, lis_id, first_name, last_name, state, party)
    print(f"Data saved to {first_name}_{last_name}_{state}_{party}_{legislator_type}_{bioguide_id}_{lis_id}.csv")