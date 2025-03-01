"""
Incremental Fusion Retail Customer Reviews Generator (Ollama Mistral Version)

This script generates realistic customer reviews for Fusion Retail products using Ollama with the Mistral model.
It saves reviews incrementally to prevent data loss in case of system crashes.

Requirements:
- requests
- pandas
- numpy
- Ollama running locally with Mistral model installed
- Existing Fusion Retail CSV files (customers.csv and transactions.csv)

Setup:
1. Install Ollama from https://ollama.ai/
2. Pull the Mistral model: `ollama pull mistral`
3. Make sure Ollama is running

Usage:
    python incremental_reviews_generator.py [--seed SEED] [--num_reviews NUM_REVIEWS] [--batch_size BATCH_SIZE]
"""

import os
import argparse
import pandas as pd
import numpy as np
import random
import time
import requests
import glob
from datetime import datetime, timedelta
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate synthetic customer reviews for Fusion Retail using Ollama Mistral')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--num_reviews', type=int, default=1000, help='Number of reviews to generate')
parser.add_argument('--batch_size', type=int, default=3, help='Number of reviews to generate per API call')
parser.add_argument('--ollama_url', type=str, default='http://localhost:11434/api/generate', help='URL for Ollama API')
parser.add_argument('--resume', action='store_true', help='Resume from previous run')
args = parser.parse_args()

# Set random seed for reproducibility
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)

# Constants
NUM_REVIEWS = args.num_reviews
BATCH_SIZE = args.batch_size
OLLAMA_URL = args.ollama_url
BATCH_SAVE_DIR = 'output/review_batches'

# Create output directory if it doesn't exist
os.makedirs(BATCH_SAVE_DIR, exist_ok=True)
os.makedirs('output', exist_ok=True)

def load_fusion_retail_data():
    """
    Load the necessary Fusion Retail data for generating reviews.
    
    Returns:
        tuple: customers_df, transactions_df
    """
    print("Loading Fusion Retail data...")
    
    try:
        customers_df = pd.read_csv('output/customers.csv')
        transactions_df = pd.read_csv('output/transactions.csv')
        
        print(f"Loaded {len(customers_df)} customers and {len(transactions_df)} transactions.")
        return customers_df, transactions_df
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the Fusion Retail data files are in the 'output' directory.")
        exit(1)

def prepare_review_data(customers_df, transactions_df, num_reviews):
    """
    Prepare the data needed for generating reviews by sampling from transactions.
    
    Args:
        customers_df (pandas.DataFrame): Customer data
        transactions_df (pandas.DataFrame): Transaction data
        num_reviews (int): Number of reviews to generate
        
    Returns:
        pandas.DataFrame: Data frame with transaction and customer info for reviews
    """
    print(f"Preparing data for {num_reviews} reviews...")
    
    # Sample transactions (ensuring we don't have more reviews than transactions)
    max_samples = min(num_reviews, len(transactions_df))
    
    # Sample from transactions, giving higher weight to more recent transactions
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    # Calculate days since transaction (relative to the most recent transaction)
    max_date = transactions_df['transaction_date'].max()
    transactions_df['days_since'] = (max_date - transactions_df['transaction_date']).dt.days
    
    # Calculate weights (more recent transactions get higher weights)
    max_days = transactions_df['days_since'].max()
    transactions_df['weight'] = 1 - (transactions_df['days_since'] / (max_days * 1.5))
    
    # Ensure weights are positive
    transactions_df['weight'] = transactions_df['weight'].clip(lower=0.1)
    
    # Normalize weights
    transactions_df['weight'] = transactions_df['weight'] / transactions_df['weight'].sum()
    
    # Handle any NaN values in weights
    transactions_df['weight'] = transactions_df['weight'].fillna(1.0/len(transactions_df))
    
    # Sample transactions
    sampled_indices = np.random.choice(
        transactions_df.index, 
        size=max_samples, 
        replace=False, 
        p=transactions_df['weight']
    )
    
    sampled_transactions = transactions_df.loc[sampled_indices]
    
    # Merge with customer data
    review_data = pd.merge(
        sampled_transactions,
        customers_df[['customer_id', 'full_name', 'age', 'gender']],
        on='customer_id',
        how='left'
    )
    
    # Calculate review date (a random period after transaction date)
    review_data['days_to_review'] = np.random.randint(1, 30, size=len(review_data))
    review_data['review_date'] = review_data['transaction_date'] + pd.to_timedelta(review_data['days_to_review'], unit='d')
    
    # Generate star ratings (biased toward higher ratings)
    rating_probabilities = [0.03, 0.07, 0.15, 0.35, 0.40]  # 1 to 5 stars
    review_data['rating'] = np.random.choice(
        [1, 2, 3, 4, 5], 
        size=len(review_data), 
        p=rating_probabilities
    )
    
    return review_data

def generate_review(product_name, product_category, customer_name, age, gender, rating, review_date):
    """
    Generate a single review using Ollama Mistral.
    
    Returns:
        tuple: (review_title, review_text)
    """
    # Format the prompt for a single review
    prompt = f"""Write a realistic customer review for the following purchase:

Product: {product_name} (Category: {product_category})
Customer: {customer_name} ({gender}, {age} years old)
Rating: {rating} stars
Date: {review_date.strftime('%Y-%m-%d')}

The review should have two parts:
1. A brief title (one short phrase or sentence)
2. The review text (3-5 sentences)

Make the review sound authentic. If the rating is low (1-2 stars), focus on problems or disappointments. If medium (3 stars), include both positives and negatives. If high (4-5 stars), focus on satisfaction and positive aspects.

Format your response exactly like this:
TITLE: [review title here]
REVIEW: [review text here]
"""

    try:
        # Call Ollama API with Mistral model
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "max_tokens": 500
            }
        )
        
        if response.status_code != 200:
            print(f"Error from Ollama API: {response.status_code}, {response.text}")
            return create_fallback_review(product_name, rating)
        
        # Extract the response text
        response_text = response.json().get("response", "").strip()
        
        # Parse title and review text
        title_start = response_text.find("TITLE:") 
        review_start = response_text.find("REVIEW:")
        
        if title_start >= 0 and review_start > title_start:
            # Extract the title and review
            title = response_text[title_start + 6:review_start].strip()
            review_text = response_text[review_start + 7:].strip()
            
            return title, review_text
        else:
            # Try alternative format (in case the model used a different format)
            lines = response_text.strip().split('\n')
            if len(lines) >= 2:
                # Assume first line is title, rest is review
                title = lines[0].replace("TITLE:", "").strip()
                review_text = ' '.join(lines[1:]).replace("REVIEW:", "").strip()
                return title, review_text
            else:
                # Fallback if we can't parse the response
                return create_fallback_review(product_name, rating)
            
    except Exception as e:
        print(f"Error generating review: {e}")
        return create_fallback_review(product_name, rating)

def create_fallback_review(product, rating):
    """Create a simple fallback review when generation fails."""
    if rating >= 4:
        title = f"Great {product}"
        text = f"I purchased this {product} recently and I'm satisfied with it. Works as expected and delivery was prompt."
    elif rating == 3:
        title = f"Decent {product}"
        text = f"This {product} is okay but not amazing. It does the job, but I expected a bit more for the price."
    else:
        title = f"Disappointed with {product}"
        text = f"Not happy with this purchase. The {product} didn't work as advertised and customer service was unhelpful."
    
    return title, text

def generate_batch_of_reviews(batch_data, batch_id):
    """
    Generate reviews for a batch of data using individual prompts.
    
    Args:
        batch_data (pandas.DataFrame): Batch of data for generation
        batch_id (int): Identifier for this batch
        
    Returns:
        list: Generated reviews with all data
    """
    batch_results = []
    
    for idx, row in batch_data.iterrows():
        product_name = row['product_name']
        product_category = row['product_category']
        customer_name = row['full_name']
        age = row['age'] if not pd.isna(row['age']) else "unknown age"
        gender = row['gender']
        rating = int(row['rating'])
        review_date = row['review_date']
        
        title, review_text = generate_review(
            product_name, product_category, customer_name, 
            age, gender, rating, review_date
        )
        
        # Create review record
        row_data = row.to_dict()
        row_data['review_title'] = title
        row_data['review_text'] = review_text
        row_data['review_id'] = f"rev_{len(batch_results) + 1:06d}"  # Temporary ID, will be adjusted later
        
        batch_results.append(row_data)
        
        # Small delay to avoid overloading Ollama
        time.sleep(0.3)
    
    # Save this batch to disk immediately
    save_batch_to_disk(batch_results, batch_id)
    
    return batch_results

def save_batch_to_disk(batch_results, batch_id):
    """
    Save a batch of reviews to disk immediately.
    
    Args:
        batch_results (list): List of review records
        batch_id (int): Batch identifier
    """
    batch_df = pd.DataFrame(batch_results)
    
    # Clean up columns for consistency
    columns_to_keep = [
        'review_id', 'customer_id', 'product_name', 'product_category', 'full_name',
        'transaction_date', 'review_date', 'rating', 'review_title', 'review_text'
    ]
    
    # Filter columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in batch_df.columns]
    batch_df = batch_df[columns_to_keep]
    
    # Save to disk
    batch_file = os.path.join(BATCH_SAVE_DIR, f'reviews_batch_{batch_id:06d}.csv')
    batch_df.to_csv(batch_file, index=False)
    print(f"✓ Saved batch {batch_id} with {len(batch_df)} reviews")

def find_completed_batches():
    """
    Find all previously completed batches.
    
    Returns:
        list: List of batch identifiers
    """
    batch_files = glob.glob(os.path.join(BATCH_SAVE_DIR, 'reviews_batch_*.csv'))
    batch_ids = []
    
    for file in batch_files:
        # Extract batch ID from filename
        filename = os.path.basename(file)
        try:
            batch_id = int(filename.replace('reviews_batch_', '').replace('.csv', ''))
            batch_ids.append(batch_id)
        except ValueError:
            continue
    
    return sorted(batch_ids)

def merge_all_batches():
    """
    Merge all batch files into a single CSV file.
    
    Returns:
        pandas.DataFrame: All reviews
    """
    print("Merging all review batches...")
    batch_files = glob.glob(os.path.join(BATCH_SAVE_DIR, 'reviews_batch_*.csv'))
    
    if not batch_files:
        print("No batch files found to merge.")
        return pd.DataFrame()
    
    all_reviews = []
    
    for file in sorted(batch_files):
        try:
            batch_df = pd.read_csv(file)
            all_reviews.append(batch_df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_reviews:
        return pd.DataFrame()
    
    # Concatenate all batches
    reviews_df = pd.concat(all_reviews, ignore_index=True)
    
    # Generate consistent review_ids
    reviews_df['review_id'] = [f"rev_{i+1:06d}" for i in range(len(reviews_df))]
    
    return reviews_df

def check_ollama_connection():
    """
    Check if Ollama server is running and has the Mistral model available.
    """
    print("Checking Ollama connection...")
    
    try:
        # Get list of models
        response = requests.get('http://localhost:11434/api/tags')
        
        if response.status_code != 200:
            print(f"Error connecting to Ollama: {response.status_code}, {response.text}")
            print("Please make sure Ollama is running.")
            return False
        
        models = response.json().get('models', [])
        model_names = [model.get('name', '') for model in models]
        
        if 'mistral' not in model_names and 'mistral:latest' not in model_names:
            print("Mistral model not found in Ollama. Please run 'ollama pull mistral' first.")
            return False
            
        print("Successfully connected to Ollama. Mistral model is available.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("Failed to connect to Ollama. Please make sure Ollama is running on localhost:11434.")
        return False
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False

def main():
    """Main function to run the review generation process."""
    print("Incremental Fusion Retail Customer Reviews Generator")
    print("--------------------------------------------------")
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("Exiting due to Ollama connection issues.")
        return
        
    # Load data
    customers_df, transactions_df = load_fusion_retail_data()
    
    # Check if we're resuming from a previous run
    completed_batch_ids = find_completed_batches()
    next_batch_id = max(completed_batch_ids) + 1 if completed_batch_ids else 1
    
    if args.resume and completed_batch_ids:
        print(f"Resuming from previous run. Found {len(completed_batch_ids)} completed batches.")
        
        # Count reviews in completed batches
        completed_reviews = 0
        for batch_id in completed_batch_ids:
            batch_file = os.path.join(BATCH_SAVE_DIR, f'reviews_batch_{batch_id:06d}.csv')
            if os.path.exists(batch_file):
                batch_df = pd.read_csv(batch_file)
                completed_reviews += len(batch_df)
        
        # Determine how many more to generate
        remaining_reviews = max(0, NUM_REVIEWS - completed_reviews)
        print(f"Already generated {completed_reviews} reviews. Need {remaining_reviews} more.")
        
        if remaining_reviews <= 0:
            print("All requested reviews have already been generated. Merging batches...")
            full_reviews_df = merge_all_batches()
            
            # Save to CSV
            output_file = 'output/customer_reviews_complete.csv'
            full_reviews_df.to_csv(output_file, index=False)
            
            print(f"\nMerged {len(full_reviews_df)} customer reviews and saved to {output_file}")
            return
    else:
        # Clean previous batch files if not resuming
        if not args.resume and completed_batch_ids:
            print(f"Found {len(completed_batch_ids)} batches from previous run.")
            answer = input("Do you want to delete previous batches and start fresh? (y/n): ")
            if answer.lower() == 'y':
                for batch_id in completed_batch_ids:
                    batch_file = os.path.join(BATCH_SAVE_DIR, f'reviews_batch_{batch_id:06d}.csv')
                    if os.path.exists(batch_file):
                        os.remove(batch_file)
                print("Deleted previous batch files.")
                next_batch_id = 1  # Reset batch numbering
                completed_batch_ids = []
                remaining_reviews = NUM_REVIEWS
            else:
                print("Will append to existing batches.")
                # Count existing reviews
                completed_reviews = 0
                for batch_id in completed_batch_ids:
                    batch_file = os.path.join(BATCH_SAVE_DIR, f'reviews_batch_{batch_id:06d}.csv')
                    if os.path.exists(batch_file):
                        batch_df = pd.read_csv(batch_file)
                        completed_reviews += len(batch_df)
                remaining_reviews = max(0, NUM_REVIEWS - completed_reviews)
        else:
            remaining_reviews = NUM_REVIEWS
    
    # Prepare review data for remaining reviews
    if remaining_reviews > 0:
        review_data = prepare_review_data(customers_df, transactions_df, remaining_reviews)
        
        # Process in batches
        num_batches = (len(review_data) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Generating {remaining_reviews} reviews in {num_batches} batches of size {BATCH_SIZE}...")
        
        try:
            batch_id = next_batch_id
            for i in tqdm(range(0, len(review_data), BATCH_SIZE)):
                batch_data = review_data.iloc[i:i+BATCH_SIZE].copy()
                
                # Generate reviews for this batch and save immediately
                generate_batch_of_reviews(batch_data, batch_id)
                
                batch_id += 1
                
                # Sleep to avoid overloading Ollama
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nGeneration interrupted. Progress has been saved.")
        except Exception as e:
            print(f"\nError during generation: {e}")
            print("Progress up to this point has been saved.")
    
    # Merge all batches
    print("\nMerging all batches to create final output file...")
    full_reviews_df = merge_all_batches()
    
    # Save to CSV
    output_file = 'output/customer_reviews_complete.csv'
    full_reviews_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated and merged {len(full_reviews_df)} customer reviews and saved to {output_file}")

if __name__ == "__main__":
    main()