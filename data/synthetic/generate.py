from anthropic import Anthropic
from dotenv import load_dotenv
import pandas as pd
import os
import time
import json
import re

load_dotenv()
client = Anthropic()

CATEGORIES = ['biling', 'technical', 'account', 'general']
URGENCY = ['low', 'medium', 'high']
TEAMS = ['biling-team', 'tech-support', 'account-team', 'general-support']

DEPARTMENT_TO_CATEGORY = {
    'Technical Support': 'technical',
    'Billing and Payments': 'biling',
    'Service Outages and Maintenance': 'account',
    'IT Support': 'technical',
    'Product Support': 'technical',
    'Customer Service': 'account',
}

CATEGORY_TO_TEAM = {
    'technical': 'tech-support',
    'biling': 'biling-team',
    'account': 'account-team',
    'general': 'general-support',
}


def parse_json_array(raw: str) -> list[str]:
    """Parsea un array JSON de strings tolerando markdown fences y texto extra."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def process_real_tickets(df: pd.DataFrame) -> list[dict]:
    """Procesa tickets del CSV. Filtra departamentos no mapeados (Returns and Exchanges,
    Human Resources, Sales and Pre-Sales, General Inquiry) porque sus textos están
    contaminados con contenido técnico mal etiquetado."""
    processed = []
    for _, row in df.iterrows():
        department = str(row.get('Department', 'None'))
        category = DEPARTMENT_TO_CATEGORY.get(department)
        if category is None:
            continue
        urgency = str(row.get('Priority', 'None'))
        team = CATEGORY_TO_TEAM[category]
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


def find_underrepresented_cases(dataset: list[dict], min_count: int = 1500) -> list[tuple]:
    counts = {}
    for ticket in dataset:
        key = (ticket['category'], ticket['urgency'])
        counts[key] = counts.get(key, 0) + 1
    targets = [
        (cat, urg) for cat in CATEGORIES for urg in URGENCY
        if counts.get((cat, urg), 0) < min_count
    ]
    return targets


IT_CASE_DESCRIPTIONS = {
    ('biling', 'low'): 'General billing questions, invoice clarification needed',
    ('biling', 'medium'): 'Billing discrepancies, unclear charges, invoice questions',
    ('biling', 'high'): 'Duplicate charges, unexpected fees, refund requests',
    ('account', 'low'): 'Profile updates, password reset, basic account questions',
    ('account', 'medium'): 'Account access issues, locked accounts, role changes',
    ('account', 'high'): 'Account compromised, unauthorized access, urgent security',
    ('technical', 'low'): 'Minor product usage questions, small UI quirks',
    ('technical', 'medium'): 'Recurring errors, integration issues, individual blockers',
    ('technical', 'high'): 'Outages, data loss, production-critical bugs',
}

GENERAL_CASE_DESCRIPTIONS = {
    'low': 'Casual product questions, feedback, suggestions, onboarding curiosity',
    'medium': 'HR questions, sales/pricing inquiries, vague but non-blocking requests',
    'high': 'Time-sensitive non-technical asks: contract questions, policy escalations',
}


def build_it_prompt(category: str, urgency: str, n: int) -> str:
    description = IT_CASE_DESCRIPTIONS.get((category, urgency), f'{category} issue with {urgency} urgency')
    return f"""Generate {n} realistic IT support tickets. Return ONLY a valid JSON array of strings, no other text, no markdown.
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

Return ONLY a valid JSON array of strings:
["ticket 1", "ticket 2", "ticket 3"]"""


def build_general_prompt(urgency: str, n: int) -> str:
    """Prompt aislado para 'general'. NO menciona IT support ni jerga técnica:
    el modelo anterior contaminaba 'general' con tickets técnicos por seguir
    instrucciones contradictorias."""
    description = GENERAL_CASE_DESCRIPTIONS[urgency]
    return f"""Generate {n} realistic customer support inquiries that do NOT fit a technical, billing, or account-management category.

These are GENERAL inquiries: feedback, product questions, HR, sales/pre-sales, onboarding, contract clarifications, miscellaneous suggestions, or vague non-actionable requests.

Urgency level: {urgency}
Examples of intent: {description}

Strict requirements:
- Each message is a single customer message (1-3 sentences)
- DO NOT include software names, error codes, hardware models, IP addresses, stack traces, or any technical jargon
- DO NOT make them about bugs, outages, login problems, or billing disputes
- Vary tone: polite, neutral, frustrated, curious
- Some should have typos or grammar issues
- They should sound like real human messages a non-technical customer would send

Return ONLY a valid JSON array of strings, no markdown, no preamble:
["inquiry 1", "inquiry 2", "inquiry 3"]"""


def generate_synthetic(targets: list[tuple], n_per_case: int = 300, batch_size: int = 50) -> list[dict]:
    """Genera tickets sintéticos. Llama varias veces con n=batch_size para evitar truncamiento.
    Usa prompts distintos: uno general (sin jerga técnica) para 'general', otro IT para el resto."""
    synthetic = []
    for category, urgency in targets:
        remaining = n_per_case
        produced_for_pair = 0
        while remaining > 0:
            n_call = min(batch_size, remaining)
            prompt = (
                build_general_prompt(urgency, n_call)
                if category == 'general'
                else build_it_prompt(category, urgency, n_call)
            )
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=4000,
                messages=[{'role': 'user', 'content': prompt}],
            )
            tickets = parse_json_array(response.content[0].text)
            if not tickets:
                print(f'  WARN: parsing failed for ({category}, {urgency}), skipping batch')
                remaining -= n_call
                continue
            for text in tickets:
                synthetic.append({
                    'text': text,
                    'category': category,
                    'urgency': urgency,
                    'team': CATEGORY_TO_TEAM[category],
                })
            produced_for_pair += len(tickets)
            remaining -= n_call
            time.sleep(1)
        print(f'  ({category}, {urgency}): {produced_for_pair} sintéticos')
    print(f'Generated {len(synthetic)} synthetic tickets')
    return synthetic


def save_dataset(dataset: list[dict], filepath: str = 'data/processed/dataset.json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f'Dataset saved to {filepath}')


def main():
    print('Loading tickets...')
    df = pd.read_csv('data/raw/IT_Support_Ticket Data.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    print('Processing real tickets')
    tickets_real = process_real_tickets(df)

    print('Finding underrepresented cases (< 1500)')
    targets = find_underrepresented_cases(tickets_real, min_count=1500)
    for cat, urg in targets:
        print(f'  underrepresented: ({cat}, {urg})')

    print('Generating synthetic tickets')
    tickets_synthetic = generate_synthetic(targets, n_per_case=300, batch_size=50)

    final_dataset = tickets_real + tickets_synthetic
    print(f'Final dataset: {len(final_dataset)} tickets')
    save_dataset(final_dataset)


if __name__ == '__main__':
    main()
