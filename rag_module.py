import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import requests
import time

# ----------------------------------------------------------------------
# 0. הגדרות וקבועים
# ----------------------------------------------------------------------
OUT_DIR = 'out'
RAG_DOCS_DIR = 'rag_docs'
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
# The API key will be provided by the environment via the GEMINI_API_KEY environment variable
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="

# Experimental settings for the Lead Conversion task
LEAD_PROFILES = {
    'Young_High_Intent': {
        'age': 30,
        'region': 'Central',
        'needs': 'Affordability and Flexibility',
        'objections': 'It is too expensive',
        'query': "Create a 3-step conversion plan for a 30-year-old lead from the Central region, focused on affordability, who objects to the price."
    },
    'Family_Security_Focused': {
        'age': 45,
        'region': 'North',
        'needs': 'Comprehensive Family Security',
        'objections': 'I need to speak to my spouse/partner',
        'query': "Create a 3-step conversion plan for a 45-year-old lead from the North region, focused on family security, who needs spousal approval."
    },
    'Senior_Asset_Protection': {
        'age': 60,
        'region': 'South',
        'needs': 'Asset Protection and Low Risk',
        'objections': 'I don\'t need it now',
        'query': "Create a 3-step conversion plan for a 60-year-old lead from the South region, focused on asset protection, who claims the timing is wrong."
    }
}

# Global variables for TF-IDF models and documents
vectorizer_lapse, tfidf_matrix_lapse, doc_ids_lapse = None, None, None
vectorizer_lead, tfidf_matrix_lead, doc_ids_lead = None, None, None
all_documents = {}

# ----------------------------------------------------------------------
# 1. RAG Document Management and TF-IDF
# ----------------------------------------------------------------------

def load_rag_documents() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Loads RAG markdown documents from rag_docs/ and extracts the Doc ID."""
    documents = {}
    doc_paths = glob.glob(os.path.join(RAG_DOCS_DIR, '*.md'))

    if not doc_paths:
        print(f"Error: No .md files found in the directory: {RAG_DOCS_DIR}. Assuming mocked RAG will run.")
        return {}, {}, {}


    for path in doc_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                fallback_id = os.path.basename(path).replace('.md', '')
                doc_id = fallback_id

                # Simple attempt to extract Doc ID from the content body
                lines = content.split('\n')
                for line in lines:
                    if '[Doc' in line:
                        try:
                            doc_id = 'Doc ' + line.split('[Doc')[1].split(']')[0].strip()
                            break
                        except IndexError:
                            continue

                documents[doc_id] = content
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            continue

    # Separate the corpora based on the Doc ID prefix
    lapse_docs = {k: v for k, v in documents.items() if k.startswith('Doc L')}
    lead_docs = {k: v for k, v in documents.items() if k.startswith('Doc C')}

    return documents, lapse_docs, lead_docs

def initialize_tfidf(docs: Dict[str, str]) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    """Trains a TF-IDF model on the document content."""
    doc_ids = list(docs.keys())
    corpus = list(docs.values())

    if not corpus:
        return None, None, []

    # Use TfidfVectorizer with English stop words for better focus on keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix, doc_ids

def retrieve_top_k_docs(query: str, vectorizer: TfidfVectorizer, tfidf_matrix: np.ndarray, doc_ids: List[str], k: int = 3) -> Dict[str, str]:
    """Performs TF-IDF retrieval using cosine similarity."""

    global all_documents

    # Handle case where TF-IDF was not initialized (e.g., no documents found)
    if vectorizer is None or tfidf_matrix is None:
        if 'Conversion' in query:
            return {"Doc C1 (Mock)": "Standard conversion strategy: offer a free consultation and 10% first-year discount.",
                    "Doc C2 (Mock)": "Objection handling guide: use testimonials for trust.",
                    "Doc C3 (Mock)": "Regional campaign: target Central region with family packages."}
        else: # Lapse query
            return {"Doc L1 (Mock)": "Retention strategy for High Risk: immediate agent call and payment flexibility.",
                    "Doc L2 (Mock)": "Retention strategy for Low Risk: automated check-in and loyalty bonus.",
                    "Doc L3 (Mock)": "Retention offer matrix: use 5% discount for customers with recent complaints."}


    # Vectorize the query
    query_vec = vectorizer.transform([query])

    # Calculate cosine similarity between the query and all documents
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get the indices of the top k documents
    top_indices = similarity_scores.argsort()[-k:][::-1]

    # Return a dictionary of {Doc ID: Content} for the top documents
    return {doc_ids[idx]: all_documents[doc_ids[idx]] for idx in top_indices}

# ----------------------------------------------------------------------
# 2. Gemini API Interaction
# ----------------------------------------------------------------------

def call_gemini_api(system_prompt: str, user_query: str, max_retries: int = 5) -> str:
    """
    Calls the Gemini API (without search grounding).
    Implements Exponential Backoff for resilience.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("⚠️ WARNING: API Key not found. Returning mocked LLM response.")
        # --- Fallback/Mock Response Logic (Ensures plan is returned even without API) ---
        if 'Predicted Lapse Probability' in user_query:
            proba_match = user_query.split('Predicted Lapse Probability (Crucial Factor): ')[1].split('\n')[0].strip()
            risk_match = user_query.split('Risk Level: ')[1].split('\n')[0].strip()
            return (
                f"LLM MOCK RESPONSE (Risk: {risk_match}, Proba: {proba_match}):\n"
                "1. Initiate Agent Outreach immediately to discuss payment flexibility [Doc L1].\n"
                "2. Offer a 5% loyalty discount on the next premium cycle to increase retention incentive [Doc L3].\n"
                "3. Schedule a follow-up call after the grace period to ensure the customer feels supported [Doc L2]."
            )
        else:
            needs_match = user_query.split('Primary Needs: ')[1].split('\n')[0].strip()
            return (
                f"LLM MOCK RESPONSE (Needs: {needs_match}):\n"
                "1. Focus messaging on the specific value proposition that addresses the lead's primary needs [Doc C1].\n"
                "2. Utilize a discount guideline (e.g., 1-month trial) to overcome initial price objections [Doc C2].\n"
                "3. Implement a short, high-touch follow-up cadence to close the deal quickly [Doc C3]."
            )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # Exponential Backoff implementation
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}{api_key}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Extract the text content
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response text.')
            return text

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"CRITICAL: Failed to call Gemini API after {max_retries} attempts: {e}")
                return f"LLM Failure: {e}"

            # Calculate wait time: 1s, 2s, 4s, ... + jitter
            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
            time.sleep(wait_time)

    return "LLM Failed after all retries."

# ----------------------------------------------------------------------
# 3. Prompt Generation and RAG Integration
# ----------------------------------------------------------------------

def generate_lapse_plan(customer_data: pd.Series, retrieved_docs: Dict[str, str]) -> str:
    """
    Generates the prompt and plan for the Lapse Prevention branch.
    Satisfies Anti-Triviality Requirement #4: Probability-in-Prompt.
    """

    # 1. Build the RAG Context from retrieved documents
    context_sections = "\n\n-- RAG Context Documents --\n\n"
    for doc_id, content in retrieved_docs.items():
        body = '\n'.join(content.split('\n')[1:]).strip()
        context_sections += f"[{doc_id}]:\n{body}\n\n"

    # 2. Define the System Instruction (Persona and Formatting)
    system_prompt = (
        "You are a Senior Retention Strategist. Your task is to analyze the customer profile and the provided "
        "policy documents (RAG Context) to devise a short, actionable, 3-step retention plan."
        "The plan MUST be generated in English, and each step MUST be grounded and end with a citation to the relevant document ID (e.g., [Doc L1])."
        "Do not invent new information. Do not mention the RAG context outside the plan."
    )

    # 3. Construct the User Query (including ML prediction)
    customer_proba = customer_data['lapse_proba']

    user_query = f"""
    Customer Profile:
    - Policy ID: {customer_data['policy_id']}
    - Risk Level: {customer_data['risk_level']}
    - Predicted Lapse Probability (Crucial Factor): {customer_proba:.2%}
    - Age: {int(customer_data['age'])}
    - Region: {customer_data['region']}
    - Monthly Premium: ${customer_data['premium']:.2f}
    - Recent Complaints (3m): {int(customer_data.get('recent_complaints', 0))}
    - Status Context: {customer_data.get('policy_status', 'N/A')}

    Based on the profile and the high predicted probability, develop the best 3-step retention action plan.
    Focus the plan on the customer's demographics, risk level, and the specific retention strategies available in the documents.

    {context_sections}

    Retention Action Plan (3 Steps):
    1.
    2.
    3.
    """

    llm_response = call_gemini_api(system_prompt, user_query)
    return llm_response

def generate_lead_plan(profile_data: Dict, retrieved_docs: Dict[str, str]) -> str:
    """Generates the prompt and plan for the Lead Conversion branch."""

    # 1. Build the RAG Context from retrieved documents
    context_sections = "\n\n-- RAG Context Documents --\n\n"
    for doc_id, content in retrieved_docs.items():
        body = '\n'.join(content.split('\n')[1:]).strip()
        context_sections += f"[{doc_id}]:\n{body}\n\n"

    # 2. Define the System Instruction (Persona and Formatting)
    system_prompt = (
        "You are a Senior Lead Conversion Manager. Your task is to analyze the lead profile and the provided "
        "policy documents (RAG Context) to devise a short, actionable, 3-step lead conversion plan."
        "The plan MUST be generated in English, and each step MUST be grounded and end with a citation to the relevant document ID (e.g., [Doc C1])."
        "Do not invent new information. Do not mention the RAG context outside the plan."
    )

    # 3. Construct the User Query
    user_query = f"""
    Lead Profile:
    - Age: {profile_data['age']}
    - Region: {profile_data['region']}
    - Primary Needs: {profile_data['needs']}
    - Main Objection: {profile_data['objections']}

    Based on the profile, develop the best 3-step lead conversion plan.
    Focus on addressing the customer's needs, overcoming the objection, and using the available conversion strategies in the documents.

    {context_sections}

    Lead Conversion Plan (3 Steps):
    1.
    2.
    3.
    """

    llm_response = call_gemini_api(system_prompt, user_query)
    return llm_response

# ----------------------------------------------------------------------
# 4. Main RAG Module Execution
# ----------------------------------------------------------------------

def initialize_rag():
    """Initializes and trains all RAG models (TF-IDF)."""
    global vectorizer_lapse, tfidf_matrix_lapse, doc_ids_lapse
    global vectorizer_lead, tfidf_matrix_lead, doc_ids_lead
    global all_documents

    # Load all documents
    all_documents, lapse_documents, lead_documents = load_rag_documents()

    # Train TF-IDF for Lapse Corpus
    vectorizer_lapse, tfidf_matrix_lapse, doc_ids_lapse = initialize_tfidf(lapse_documents)

    # Train TF-IDF for Lead Corpus
    vectorizer_lead, tfidf_matrix_lead, doc_ids_lead = initialize_tfidf(lead_documents)

    print("🧠 RAG Initialized. TF-IDF models trained on Lapse and Lead corpora.")


def rag_analysis(df_rag: pd.DataFrame, metrics: dict):
    """
    RAG (Generation Augmented Retrieval) module - Executes the two required RAG tasks.
    """

    # 1. Initialization
    if not all_documents and os.path.isdir(RAG_DOCS_DIR):
        initialize_rag()

    # 2. Select RAG Characters for Lapse Prevention (High, Median, Low Risk)
    def select_rag_characters(df_rag: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df_rag.sort_values(by='lapse_proba', ascending=False).reset_index(drop=True)

        # Select High, Median, Low risk customers
        high_risk = df_sorted.iloc[0].to_frame().T.copy()
        median_index = int(len(df_sorted) * 0.5)
        median_risk = df_sorted.iloc[median_index].to_frame().T.copy()
        low_risk = df_sorted.iloc[-1].to_frame().T.copy()

        # --- FIX: Inject synthetic RAG context features based on risk level ---
        # This solves the KeyError by relying on features known to exist ('age', 'premium')
        # and adding new, mocked features for richer LLM context.

        high_risk['risk_level'] = 'High'
        high_risk['recent_complaints'] = 2
        high_risk['policy_status'] = 'High Coverage / High Premium'

        median_risk['risk_level'] = 'Median'
        median_risk['recent_complaints'] = 0
        median_risk['policy_status'] = 'Stable / Mid-level Premium'

        low_risk['risk_level'] = 'Low'
        low_risk['recent_complaints'] = 0
        low_risk['policy_status'] = 'Long-Term / Low Premium'

        rag_characters = pd.concat([high_risk, median_risk, low_risk])

        # Use features guaranteed to exist or newly created
        RAG_COLS = ['risk_level', 'policy_id', 'lapse_proba', 'age', 'premium', 'region', 'lapse_next_3m', 'recent_complaints', 'policy_status']

        # Filter for columns that actually exist in the combined DataFrame
        available_cols = [col for col in RAG_COLS if col in rag_characters.columns]

        return rag_characters[available_cols].reset_index(drop=True)


    rag_characters_df = select_rag_characters(df_rag)

    # 3. Execute RAG - Lapse Prevention Branch
    lapse_output_sections = ["\n--- LAPSE PREVENTION BRANCH: RAG 3-STEP PLANS ---\n"]
    for index, customer in rag_characters_df.iterrows():
        # Create a profile-based query for retrieval
        query = (
            f"Retention strategy for {customer['risk_level']} risk policy with {customer['lapse_proba']:.2%} lapse probability. "
            f"Customer is {int(customer['age'])} years old from the {customer['region']} region, with {int(customer.get('recent_complaints', 0))} recent complaints."
        )

        # Retrieve relevant documents from the Lapse corpus
        retrieved_docs = retrieve_top_k_docs(
            query,
            vectorizer_lapse,
            tfidf_matrix_lapse,
            doc_ids_lapse,
            k=3
        )

        # Generate the plan using the retrieved documents (context)
        plan = generate_lapse_plan(customer, retrieved_docs)

        lapse_output_sections.append(
            f"*** Customer Profile: {customer['risk_level']} Risk ({customer['lapse_proba']:.2%} Proba) ***\n"
            f"Query: {query}\n"
            f"Relevant Documents: {list(retrieved_docs.keys())}\n\n"
            f"--- 3-Step Retention Plan ---\n{plan}\n\n"
        )

    # 4. Execute RAG - Lead Conversion Branch
    lead_output_sections = ["\n\n--- LEAD CONVERSION BRANCH: RAG 3-STEP PLANS ---\n"]
    for profile_name, profile_data in LEAD_PROFILES.items():
        # Use the predefined query from the LEAD_PROFILES dict for retrieval
        query = profile_data['query']

        # Retrieve relevant documents from the Lead corpus
        retrieved_docs = retrieve_top_k_docs(
            query,
            vectorizer_lead,
            tfidf_matrix_lead,
            doc_ids_lead,
            k=3
        )

        # Generate the plan using the retrieved documents (context)
        plan = generate_lead_plan(profile_data, retrieved_docs)

        lead_output_sections.append(
            f"*** Lead Profile: {profile_name} ***\n"
            f"Query: {query}\n"
            f"Relevant Documents: {list(retrieved_docs.keys())}\n\n"
            f"--- 3-Step Conversion Plan ---\n{plan}\n\n"
        )

    # 5. Save Final Output
    rag_path = os.path.join(OUT_DIR, 'rag_output.txt')
    final_output = "\n".join(lapse_output_sections + lead_output_sections)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    with open(rag_path, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print(f"\n✅ RAG Analysis Complete! Full output saved to {rag_path}")

    return final_output