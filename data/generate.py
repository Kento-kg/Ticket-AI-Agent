from anthropic import Anthropic
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
import os
import time
import json

load_dotenv()
client = Anthropic()

CATEGORIES = ['biling', 'technical', 'shipping', 'account', 'general']
URGENCY = ['low', 'medium', 'hgih', 'critical']
TEAMS = ['biling-team', 'tech-support', 'logistics', 'account-team', 'general-support']

DEPARTMENT_TO_CATEGORY = {
    'Technical Support': 'technical',
    'Billing and Payments': 'biling',
    'Service Outages and Maintenance': 'account',
    'IT Support': 'technical',
    'Human Resources': 'general',
    'Returns and Exchanges': 'shipping',
    'Sales and Pre-Sales': 'general',
    'Product Support': 'technical',
    'Customer Service': 'account',
    'General Inquiry': 'general'
}

CATEGORY_TO_TEAM = {
    'technical': 'tech-support',
    'biling': 'biling-team',
    'account': 'account-team',
    'general': 'general-support',
    'shipping': 'logistics'
}

def process_real_tickets(df: pd.DataFrame) -> list[dict]:
    processed = []
    for idx, row in df.iterrows():
        department = str(row.get('Department', 'None'))
        category = DEPARTMENT_TO_CATEGORY.get(department, 'general')
        urgency = str(row.get('Priority', 'None'))
        team = CATEGORY_TO_TEAM.get(category, 'general-support')
        ticket_text = str(row.get('Body', ''))
        if not ticket_text or len(ticket_text) < 5:
            continue
        processed.append({
            'text': ticket_text,
            'category': category,
            'urgency': urgency,
            'team': team
        })
    print(f'Processed {len(processed)} real tickets')
    return processed

def find_underrepresented_cases(dataset: list[dict]) -> list[tuple]:
    min_count = 1500
    counts = {}
    for ticket in dataset:
        key = (ticket['category'], ticket['urgency'])
        counts[key] = counts.get(key, 0) +1
    underrepresented = [key for key, count in counts.items() if count < min_count]
    return underrepresented

def generate_synthetic_underrepresented(underrepresented: list[tuple], n_per_case=100) -> list[dict]:
    synthetic = []
    case_descriptions = {
        ('shipping', 'medium'): 'Minor delays, tracking not updating',
        ('biling', 'low'): 'General billing questions, invoice clarification needed',
        ('general', 'medium'): 'General issues with some importance',
        ('general', 'high'): 'Important issues requiring immediate attention',
        ('biling', 'medium'): 'Billing discrepancies, unclear charges, invoice questions',
        ("shipping", "high"): "Package lost, significant delays, major tracking issues",
        ("billing", "high"): "Duplicate charges, unexpected fees, refund requests",
        ("shipping", "low"): "Shipping questions, method preferences",
        ("general", "low"): "General inquiries, feedback, suggestions",
    }
    for category, urgency in underrepresented:
        description = case_descriptions.get((category, urgency), f'{category} issue with {urgency} urgency')
        prompt = f"""Generate {n_per_case} realistic IT support tickets. Return ONLY a valid JSON array of strings, no other text, no markdown.
        Category: {category}
        Urgency: {urgency}
        Description: {description}

        Requirements:
        - Each ticket is a single customer message (1-3 sentences)
        - Vary the tone: frustrated, neutral, polite, urgent
        - Include realistic IT details: software names, error messages, hardware models
        - Some should have typos or grammar issues
        - Make them authentic - real support ticket language
        - Varied length and detail level

        Remember, return ONLY a valid JSON array of strings, no other text, no markdown:
        ["ticket 1", "ticket 2", "ticket 3"]       
        """
        response = client.messages.create(
            model = 'claude-sonnet-4-6',
            max_tokens = 2000,
            messages = [{'role': 'user', 'content': prompt}]
        )
        try:
            tickets_synthetic = json.loads(response.content[0].text)
        except:
            print(f"Error parsing JSON")
            tickets_list = []
        
        for text in tickets_synthetic:
            synthetic.append({
                "text": text,
                "category": category,
                "urgency": urgency,
                "team": CATEGORY_TO_TEAM[category]
            })
        time.sleep(1)
    print(f'Generated {len(synthetic)} synthetic tickets')
    return synthetic        

def save_dataset(dataset: list[dict], filepath: str='processed/dataset.json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f'Dataset saved ti {filepath}')

def main():
    print('Loading tickets...')
    df = pd.read_csv('raw/IT_Support_Ticket Data.csv')
    df = df.drop(columns=['Unnamed: 0'])
    print('Processing real tickets')
    tickets_real = process_real_tickets(df)
    print('Finding underrepresented cases')
    underrepresented = find_underrepresented_cases(tickets_real)
    print('Generating synthetic tickets for underrepresented edge cases')
    tickets_synthetic = []
    if underrepresented:
        tickets_synthetic = generate_synthetic_underrepresented(
            underrepresented,
            n_per_case=1
        )
    final_dataset = tickets_real + tickets_synthetic
    print('Saving dataset')
    save_dataset(final_dataset)

if __name__ == '__main__':
    main()
    