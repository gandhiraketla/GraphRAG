"""
Fusion Retail Customer Reviews Generator

This script generates realistic customer reviews for Fusion Retail products using OpenAI's GPT model.
It connects reviews to existing customers and products from the previously generated datasets.

Requirements:
- OpenAI Python package (v1.0.0 or higher)
- pandas
- numpy
- Existing Fusion Retail CSV files (customers.csv and transactions.csv)

Usage:
    python fusion_retail_reviews_generator.py [--seed SEED] [--num_reviews NUM_REVIEWS] [--batch_size BATCH_SIZE]

Options:
    --seed SEED                Random seed for reproducibility (default: 42)
    --num_reviews NUM_REVIEWS  Number of reviews to generate (default: 1000)
    --batch_size BATCH_SIZE    Number of reviews to generate per API call (default: 10)
"""

import os
import argparse
import pandas as pd
import numpy as np
import random
import json
import time
from datetime import datetime, timedelta
from openai import OpenAI
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate synthetic customer reviews for Fusion Retail')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--num_reviews', type=int, default=1000, help='Number of reviews to generate')
parser.add_argument('--batch_size', type=int, default=10, help='Number of reviews to generate per API call')
parser.add_argument('--api_key', type=str, help='OpenAI API key')
args = parser.parse_args()

# Set random seed for reproducibility
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)

# Set OpenAI API key
if args.api_key:
    api_key = args.api_key
else:
    # Try to get from environment variable
    api_key = "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"
    if not api_key:
        raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or use the --api_key argument.")

# Constants
NUM_REVIEWS = 500
BATCH_SIZE = args.batch_size

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

def create_prompt_for_batch(batch_data):
    """
    Create a prompt for generating a batch of reviews.
    
    Args:
        batch_data (pandas.DataFrame): Batch of data for review generation
        
    Returns:
        str: Prompt for the language model
    """
    reviews_instructions = []
    
    for _, row in batch_data.iterrows():
        product_name = row['product_name']
        product_category = row['product_category']
        customer_name = row['full_name']
        customer_age = row['age'] if not pd.isna(row['age']) else "unknown age"
        customer_gender = row['gender']
        rating = int(row['rating'])
        
        # Create instruction for this specific review
        instruction = (
            f"- Product: {product_name} (Category: {product_category})\n"
            f"  Customer: {customer_name} ({customer_gender}, {customer_age} years old)\n"
            f"  Rating: {rating} stars\n"
            f"  Date: {row['review_date'].strftime('%Y-%m-%d')}\n"
        )
        
        reviews_instructions.append(instruction)
    
    instructions = "\n".join(reviews_instructions)
    
    prompt = f"""Generate {len(batch_data)} realistic customer reviews for products purchased from Fusion Retail, an omnichannel retailer specializing in consumer electronics and home goods.

For each review, I'll provide the product name, product category, customer details, star rating, and review date. Please generate an authentic-sounding review that matches these details.

Please output the reviews as a JSON array, where each review is a JSON object with the following fields:
- review_text: The text of the review
- review_title: A brief title for the review (1-10 words)

Here are the reviews to generate:

{instructions}

Please make the reviews varied in length, tone, and style. The reviews should sound like they were written by real people and mention specific features or experiences with the products. Include some spelling or grammar mistakes in approximately 15% of reviews to make them more authentic. 

For lower ratings (1-2 stars), focus on product issues, disappointments, or service problems. For medium ratings (3 stars), include mixed experiences. For higher ratings (4-5 stars), emphasize positive experiences and satisfaction.

Respond ONLY with the JSON array of reviews.
"""
    return prompt

def generate_reviews_batch(batch_data, client):
    """
    Generate a batch of reviews using OpenAI's GPT model.
    
    Args:
        batch_data (pandas.DataFrame): Batch of data for review generation
        client (OpenAI): OpenAI client instance
        
    Returns:
        list: Generated reviews with titles and text
    """
    prompt = create_prompt_for_batch(batch_data)
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Use the OpenAI client API (v1.0.0+)
            response = client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if preferred
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates realistic customer reviews."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                n=1,
                stop=None
            )
            
            # Extract the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Find JSON array in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                
                try:
                    reviews_data = json.loads(json_str)
                    return reviews_data
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON, retrying ({attempt+1}/{max_retries})...")
            else:
                print(f"No valid JSON found in response, retrying ({attempt+1}/{max_retries})...")
            
        except Exception as e:
            print(f"Error in API call ({attempt+1}/{max_retries}): {e}")
        
        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    # If all retries fail, return empty list
    print("Failed to generate reviews after multiple attempts")
    return []

def generate_all_reviews(review_data, batch_size=10):
    """
    Generate all reviews in batches.
    
    Args:
        review_data (pandas.DataFrame): Data for all reviews
        batch_size (int): Number of reviews per batch
        
    Returns:
        pandas.DataFrame: All generated reviews
    """
    all_reviews = []
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Process in batches
    num_batches = (len(review_data) + batch_size - 1) // batch_size
    print(f"Generating reviews in {num_batches} batches of size {batch_size}...")
    
    for i in tqdm(range(0, len(review_data), batch_size)):
        batch_data = review_data.iloc[i:i+batch_size].copy()
        
        # Generate reviews for this batch
        batch_reviews = generate_reviews_batch(batch_data, client)
        
        # If generation failed, create dummy reviews
        if not batch_reviews:
            batch_reviews = [{"review_title": "Error generating review", 
                             "review_text": "There was an error generating this review."} 
                             for _ in range(len(batch_data))]
        
        # Combine the original data with the generated reviews
        for j, review_dict in enumerate(batch_reviews):
            if j < len(batch_data):  # Safety check
                row_data = batch_data.iloc[j].to_dict()
                
                # Add the generated review content
                row_data['review_title'] = review_dict.get('review_title', 'No Title')
                row_data['review_text'] = review_dict.get('review_text', 'No Review Text')
                
                all_reviews.append(row_data)
        
        # Sleep to avoid hitting API rate limits
        time.sleep(1)
    
    # Convert to DataFrame
    reviews_df = pd.DataFrame(all_reviews)
    
    # Clean up columns
    columns_to_keep = [
        'customer_id', 'product_name', 'product_category', 'full_name',
        'transaction_date', 'review_date', 'rating', 'review_title', 'review_text'
    ]
    
    # Filter columns that exist (some might be missing if there were errors)
    columns_to_keep = [col for col in columns_to_keep if col in reviews_df.columns]
    
    # Generate review_id
    reviews_df['review_id'] = [f"rev_{i+1:06d}" for i in range(len(reviews_df))]
    
    # Reorder columns
    final_columns = ['review_id'] + columns_to_keep
    reviews_df = reviews_df[final_columns]
    
    return reviews_df

def main():
    """Main function to run the review generation process."""
    print("Fusion Retail Customer Reviews Generator")
    print("---------------------------------------")
    
    # Load data
    customers_df, transactions_df = load_fusion_retail_data()
    
    # Prepare review data
    review_data = prepare_review_data(customers_df, transactions_df, NUM_REVIEWS)
    
    # Generate reviews
    reviews_df = generate_all_reviews(review_data, BATCH_SIZE)
    
    # Save to CSV
    output_file = 'output/customer_reviews.csv'
    reviews_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated {len(reviews_df)} customer reviews and saved to {output_file}")

if __name__ == "__main__":
    main()