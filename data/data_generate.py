"""
Fusion Retail Synthetic Data Generator

This script generates realistic synthetic data for a fictitious omnichannel retailer, Fusion Retail.
It creates five CSV files:
1. customers.csv - Customer profile data
2. transactions.csv - Customer transaction data
3. campaigns.csv - Marketing campaign data
4. interactions.csv - Customer interaction data
5. support_tickets.csv - Customer support ticket data

The data includes realistic relationships between entities and incorporates
patterns and distributions that mimic real-world retail operations.

Usage:
    python fusion_retail_data_generator.py [--seed SEED] [--scale SCALE]

Options:
    --seed SEED    Random seed for reproducibility (default: 42)
    --scale SCALE  Scale factor for data volume (default: 1.0)
                   For example, 0.5 generates half the data, 2.0 doubles it

"""

import pandas as pd
import numpy as np
import random
import argparse
import os
from datetime import datetime, timedelta
from faker import Faker
import uuid
import math

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate synthetic data for Fusion Retail')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for data volume')
args = parser.parse_args()

# Set random seed for reproducibility
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
faker = Faker()
Faker.seed(random_seed)

# Scale factors for each dataset
scale_factor = args.scale
NUM_CUSTOMERS = int(5000 * scale_factor)
NUM_TRANSACTIONS = int(50000 * scale_factor)
NUM_CAMPAIGNS = int(200 * scale_factor)
NUM_INTERACTIONS = int(100000 * scale_factor)
NUM_SUPPORT_TICKETS = int(3000 * scale_factor)

# Date range for the data (last 5 years)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)

print(f"Generating data with seed {random_seed} and scale factor {scale_factor}")

def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

def generate_customers(num_records):
    """
    Generate customer profile data.
    
    Args:
        num_records (int): Number of customer records to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing customer data
    """
    print(f"Generating {num_records} customer records...")
    
    # Age distribution skewed toward 25-45 age range
    age_distribution = np.clip(np.random.normal(35, 12, num_records), 18, 80).astype(int)
    
    # Lists for generating realistic data
    genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
    gender_weights = [0.48, 0.48, 0.03, 0.01]
    preferred_channels = ['online', 'in-store', 'both']
    channel_weights = [0.45, 0.25, 0.3]  # Skewed towards online and both
    
    # State distribution (top 15 states by population)
    states = {
        'CA': 'California', 'TX': 'Texas', 'FL': 'Florida', 'NY': 'New York', 
        'PA': 'Pennsylvania', 'IL': 'Illinois', 'OH': 'Ohio', 'GA': 'Georgia', 
        'NC': 'North Carolina', 'MI': 'Michigan', 'NJ': 'New Jersey', 
        'VA': 'Virginia', 'WA': 'Washington', 'AZ': 'Arizona', 'MA': 'Massachusetts'
    }
    state_weights = [0.12, 0.09, 0.07, 0.06, 0.04, 0.04, 0.035, 0.035, 
                     0.03, 0.03, 0.025, 0.025, 0.02, 0.02, 0.02]
    # Normalize state weights to ensure they sum to 1.0
    state_weights = [w/sum(state_weights) for w in state_weights]
    
    # Cities per state (simplified)
    cities_by_state = {
        'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'San Jose'],
        'TX': ['Houston', 'Austin', 'Dallas', 'San Antonio', 'Fort Worth'],
        'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Tallahassee'],
        'NY': ['New York City', 'Buffalo', 'Rochester', 'Syracuse', 'Albany'],
        'PA': ['Philadelphia', 'Pittsburgh', 'Harrisburg', 'Allentown', 'Erie'],
        'IL': ['Chicago', 'Springfield', 'Peoria', 'Rockford', 'Champaign'],
        'OH': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo', 'Akron'],
        'GA': ['Atlanta', 'Savannah', 'Augusta', 'Athens', 'Macon'],
        'NC': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham', 'Winston-Salem'],
        'MI': ['Detroit', 'Grand Rapids', 'Ann Arbor', 'Lansing', 'Flint'],
        'NJ': ['Newark', 'Jersey City', 'Paterson', 'Elizabeth', 'Trenton'],
        'VA': ['Virginia Beach', 'Richmond', 'Norfolk', 'Arlington', 'Alexandria'],
        'WA': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver', 'Bellevue'],
        'AZ': ['Phoenix', 'Tucson', 'Mesa', 'Scottsdale', 'Tempe'],
        'MA': ['Boston', 'Worcester', 'Springfield', 'Cambridge', 'Lowell']
    }
    
    data = []
    customer_ids = []
    
    for i in range(num_records):
        # Generate a UUID for customer_id
        customer_id = str(uuid.uuid4())
        customer_ids.append(customer_id)
        
        # Generate name and gender
        gender = np.random.choice(genders, p=gender_weights)
        if gender == 'Male':
            first_name = faker.first_name_male()
        elif gender == 'Female':
            first_name = faker.first_name_female()
        else:
            first_name = faker.first_name()
        
        last_name = faker.last_name()
        full_name = f"{first_name} {last_name}"
        
        # Generate email based on name
        email_format = random.choice([
            f"{first_name.lower()}.{last_name.lower()}@{faker.free_email_domain()}",
            f"{first_name.lower()[0]}{last_name.lower()}@{faker.free_email_domain()}",
            f"{last_name.lower()}{first_name.lower()[0]}@{faker.free_email_domain()}",
            f"{first_name.lower()}_{last_name.lower()}@{faker.free_email_domain()}"
        ])
        email = email_format  # Assign the generated email to the email variable
        
        # Geographic information
        state_code = np.random.choice(list(states.keys()), p=state_weights)
        state_name = states[state_code]
        city = random.choice(cities_by_state[state_code])
        street_address = faker.street_address()
        zip_code = faker.zipcode()
        
        # Phone number
        phone = faker.phone_number()
        
        # Age and registration date
        age = age_distribution[i]
        registration_date = random_date(START_DATE, END_DATE)
        
        # Preferred channel - adjusted by age
        if age < 30:
            channel_weights_adjusted = [0.6, 0.1, 0.3]  # Younger prefer online
        elif age > 60:
            channel_weights_adjusted = [0.3, 0.4, 0.3]  # Older prefer in-store
        else:
            channel_weights_adjusted = channel_weights
            
        preferred_channel = np.random.choice(preferred_channels, p=channel_weights_adjusted)
        
        # Create customer record
        customer = {
            'customer_id': customer_id,
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'email': email,
            'phone': phone,
            'street_address': street_address,
            'city': city,
            'state': state_name,
            'state_code': state_code,
            'zip_code': zip_code,
            'registration_date': registration_date.strftime('%Y-%m-%d'),
            'preferred_channel': preferred_channel
        }
        
        data.append(customer)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values (approximately 2%)
    for column in ['phone', 'street_address', 'age']:
        mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
        df.loc[mask, column] = np.nan
    
    # Remove state_code before saving (it was just used for generation)
    df = df.drop('state_code', axis=1)
    
    return df, customer_ids

def generate_transactions(customer_ids, customer_df, num_records):
    """
    Generate transaction data based on customer profiles.
    
    Args:
        customer_ids (list): List of customer IDs
        customer_df (pandas.DataFrame): DataFrame containing customer data
        num_records (int): Number of transaction records to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing transaction data
    """
    print(f"Generating {num_records} transaction records...")
    
    # Product categories and their price ranges
    product_categories = {
        'Smartphones': (400, 1200),
        'Laptops': (500, 2500),
        'Tablets': (200, 800),
        'Desktop Computers': (600, 2000),
        'Computer Accessories': (20, 150),
        'TVs': (300, 3000),
        'Audio Equipment': (50, 500),
        'Gaming Consoles': (200, 600),
        'Smart Home Devices': (30, 300),
        'Kitchen Appliances': (100, 1000),
        'Small Kitchen Appliances': (25, 200),
        'Furniture': (200, 2000),
        'Home Decor': (20, 300),
        'Bedding': (30, 300),
        'Cookware': (40, 400)
    }
    
    # Products within each category
    products = {
        'Smartphones': ['iPhone 13', 'Samsung Galaxy S22', 'Google Pixel 6', 'OnePlus 10', 'Xiaomi Mi 12'],
        'Laptops': ['MacBook Pro', 'Dell XPS 15', 'HP Spectre', 'Lenovo ThinkPad', 'Asus ZenBook'],
        'Tablets': ['iPad Pro', 'Samsung Galaxy Tab', 'Amazon Fire HD', 'Microsoft Surface', 'Lenovo Tab'],
        'Desktop Computers': ['iMac', 'Dell Inspiron Desktop', 'HP Pavilion', 'Lenovo IdeaCentre', 'Asus ROG'],
        'Computer Accessories': ['Logitech Mouse', 'Mechanical Keyboard', 'USB-C Hub', 'External Hard Drive', 'Webcam'],
        'TVs': ['Samsung QLED TV', 'LG OLED TV', 'Sony Bravia', 'TCL Roku TV', 'Vizio SmartCast TV'],
        'Audio Equipment': ['Bose Headphones', 'Sony Soundbar', 'JBL Bluetooth Speaker', 'Audio-Technica Turntable', 'Sonos Speaker'],
        'Gaming Consoles': ['PlayStation 5', 'Xbox Series X', 'Nintendo Switch', 'Steam Deck', 'Oculus Quest'],
        'Smart Home Devices': ['Amazon Echo', 'Google Nest', 'Ring Doorbell', 'Philips Hue Lights', 'Smart Thermostat'],
        'Kitchen Appliances': ['Refrigerator', 'Dishwasher', 'Microwave Oven', 'Electric Range', 'Range Hood'],
        'Small Kitchen Appliances': ['Coffee Maker', 'Toaster', 'Blender', 'Air Fryer', 'Food Processor'],
        'Furniture': ['Sofa', 'Dining Table', 'Bed Frame', 'Bookshelf', 'Office Desk'],
        'Home Decor': ['Area Rug', 'Wall Art', 'Throw Pillows', 'Table Lamp', 'Curtains'],
        'Bedding': ['Comforter Set', 'Sheets', 'Pillows', 'Mattress Topper', 'Duvet Cover'],
        'Cookware': ['Cookware Set', 'Dutch Oven', 'Cast Iron Skillet', 'Knife Set', 'Baking Sheet']
    }
    
    # Store locations
    store_locations = ['Online', 'San Francisco, CA', 'Los Angeles, CA', 'New York, NY', 
                       'Chicago, IL', 'Houston, TX', 'Miami, FL', 'Seattle, WA', 
                       'Boston, MA', 'Atlanta, GA', 'Denver, CO']
    
    # Payment methods
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay', 'Cash', 'Gift Card']
    payment_weights = [0.35, 0.25, 0.15, 0.1, 0.05, 0.05, 0.05]
    
    data = []
    
    # Customer purchase frequency adjustment based on age and preferred channel
    def get_purchase_frequency(age, preferred_channel):
        # Handle NaN value in age
        if pd.isna(age):
            age = 35  # Use average age if missing
            
        base = 1.0
        # Age adjustment
        if age < 25:
            base *= 0.8  # Younger customers buy less frequently
        elif 25 <= age <= 45:
            base *= 1.5  # Prime shopping demographic
        
        # Channel adjustment
        if preferred_channel == 'both':
            base *= 1.5  # Omnichannel customers buy more
        elif preferred_channel == 'online':
            base *= 1.2  # Online shoppers also tend to buy more
        
        return base
    
    # Customer purchase preferences based on age
    def get_category_preferences(age):
        # Handle NaN value in age
        if pd.isna(age):
            age = 35  # Use average age if missing
            
        if age < 30:
            return ['Smartphones', 'Laptops', 'Gaming Consoles', 'Audio Equipment', 'Smart Home Devices']
        elif age >= 30 and age < 45:
            return ['Smartphones', 'Kitchen Appliances', 'Furniture', 'Home Decor', 'Smart Home Devices']
        else:
            return ['TVs', 'Kitchen Appliances', 'Small Kitchen Appliances', 'Furniture', 'Home Decor']
    
    # Seasonal sales patterns
    def get_seasonal_factor(date):
        month = date.month
        day = date.day
        
        # Holiday season (November-December)
        if month == 11 or month == 12:
            return 2.0
        # Black Friday
        if month == 11 and 22 <= day <= 30:
            return 3.0
        # Summer sale (July)
        if month == 7:
            return 1.5
        # Back to school (August)
        if month == 8:
            return 1.7
        # Regular season
        return 1.0
    
    # Assign transactions with weighting by customer profile
    customer_weights = []
    
    for idx, customer_id in enumerate(customer_ids):
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        age = customer['age']
        preferred_channel = customer['preferred_channel']
        registration_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        
        # Calculate weight based on frequency and time as customer
        frequency = get_purchase_frequency(age, preferred_channel)
        time_factor = (END_DATE - registration_date).days / 365.0  # Years as customer
        weight = frequency * max(time_factor, 0.1)  # Ensure new customers still get some transactions
        
        customer_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(customer_weights)
    customer_weights = [w/total_weight for w in customer_weights]
    
    # Ensure there are no NaN values in weights
    customer_weights = [w if not math.isnan(w) else 1.0/len(customer_weights) for w in customer_weights]
    
    # Ensure there are no NaN values in weights
    customer_weights = [w if not math.isnan(w) else 1.0/len(customer_weights) for w in customer_weights]
    
    # Sample customer indices based on weights
    customer_weights = np.array(customer_weights)
    
    # Handle any NaN or zero values by replacing with small positive values
    customer_weights = np.nan_to_num(customer_weights, nan=1.0/len(customer_weights))
    
    # Ensure weights sum to 1.0
    customer_weights = customer_weights / np.sum(customer_weights)
    
    customer_indices = np.random.choice(
        range(len(customer_ids)), 
        size=num_records, 
        p=customer_weights,
        replace=True
    )
    
    for i in range(num_records):
        # Get the customer for this transaction
        customer_idx = customer_indices[i]
        customer_id = customer_ids[customer_idx]
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        
        # Get customer details
        age = customer['age']
        preferred_channel = customer['preferred_channel']
        registration_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        
        # Determine transaction date (after registration)
        days_since_registration = (END_DATE - registration_date).days
        if days_since_registration <= 1:
            # For very recent registrations, set transaction on same day
            transaction_date = registration_date
        else:
            # Otherwise, generate a random date after registration
            max_days = min(days_since_registration, 365 * 5)  # Cap at 5 years or days since registration
            random_days = np.random.randint(0, max_days)
            transaction_date = registration_date + timedelta(days=random_days)
        
        # Apply seasonal factor to determine if transaction happens
        seasonal_factor = get_seasonal_factor(transaction_date)
        if random.random() > seasonal_factor * 0.5:  # Higher seasonal factor = higher chance of transaction
            continue
        
        # Determine store location based on preferred channel
        if preferred_channel == 'online':
            store_location = 'Online'
        elif preferred_channel == 'in-store':
            store_location = random.choice(store_locations[1:])  # Skip 'Online'
        else:  # 'both'
            store_location = random.choice(store_locations)
        
        # Select product category with preference based on age
        preferred_categories = get_category_preferences(age)
        if random.random() < 0.7:  # 70% chance to pick from preferred categories
            category = random.choice(preferred_categories)
        else:
            category = random.choice(list(product_categories.keys()))
        
        # Select product from category
        product_name = random.choice(products[category])
        
        # Determine price and quantity
        min_price, max_price = product_categories[category]
        base_price = round(random.uniform(min_price, max_price), 2)
        
        # Add some price variation
        price_variation = random.uniform(0.8, 1.2)
        price = round(base_price * price_variation, 2)
        
        # Quantity - most purchases are for 1 item, but allow for multiples
        quantity_dist = np.random.poisson(0.3) + 1  # Poisson with mean 0.3, shifted by 1
        quantity = min(quantity_dist, 5)  # Cap at 5 items
        
        # Apply discounts
        if random.random() < 0.3:  # 30% chance of discount
            discount_pct = random.choice([5, 10, 15, 20, 25, 30])
            discount_applied = discount_pct
        else:
            discount_applied = 0
        
        # Payment method
        payment_method = np.random.choice(payment_methods, p=payment_weights)
        
        # Create unique transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Create transaction record
        transaction = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'product_name': product_name,
            'product_category': category,
            'quantity': quantity,
            'price': price,
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'store_location': store_location,
            'payment_method': payment_method,
            'discount_applied': discount_applied
        }
        
        data.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some outliers in price and quantity
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
    for idx in outlier_indices:
        if random.random() < 0.5:
            # Price outlier
            df.at[idx, 'price'] = df.at[idx, 'price'] * random.uniform(3, 10)
        else:
            # Quantity outlier
            df.at[idx, 'quantity'] = random.randint(10, 50)
    
    # Add a few duplicate transactions (0.5%)
    duplicate_count = int(len(df) * 0.005)
    for _ in range(duplicate_count):
        idx = random.randint(0, len(df) - 1)
        duplicate = df.iloc[idx].copy()
        duplicate['transaction_id'] = str(uuid.uuid4())  # New transaction ID
        df = pd.concat([df, pd.DataFrame([duplicate])], ignore_index=True)
    
    return df

def generate_campaigns(num_records):
    """
    Generate marketing campaign data.
    
    Args:
        num_records (int): Number of campaign records to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing campaign data
    """
    print(f"Generating {num_records} marketing campaign records...")
    
    campaign_types = [
        'Email Marketing', 'Social Media', 'In-Store Promotion', 'TV Advertisement',
        'Online Display Ads', 'Search Engine Marketing', 'SMS Marketing',
        'Print Advertisement', 'Radio Advertisement', 'Influencer Marketing'
    ]
    
    target_segments = [
        'All Customers', 'New Customers', 'Loyal Customers', 'Inactive Customers',
        'Young Adults (18-25)', 'Adults (26-40)', 'Middle-aged (41-60)', 'Seniors (60+)',
        'Online Shoppers', 'In-Store Shoppers', 'High-Value Customers',
        'Technology Enthusiasts', 'Home Improvement', 'Kitchen Enthusiasts',
        'West Coast', 'East Coast', 'Midwest', 'Southern States'
    ]
    
    data = []
    
    # Campaign naming patterns
    campaign_prefixes = ['Summer', 'Winter', 'Spring', 'Fall', 'Holiday', 'Black Friday', 
                         'Back to School', 'New Year', 'Anniversary', 'Flash', 'Clearance',
                         'VIP', 'Exclusive', 'Limited Time', 'Weekend']
    
    campaign_suffixes = ['Sale', 'Special', 'Event', 'Promotion', 'Deals', 'Discount',
                         'Extravaganza', 'Bonanza', 'Blowout', 'Collection']
    
    for i in range(num_records):
        # Generate campaign ID
        campaign_id = str(uuid.uuid4())
        
        # Generate campaign name
        campaign_prefix = random.choice(campaign_prefixes)
        campaign_suffix = random.choice(campaign_suffixes)
        campaign_name = f"{campaign_prefix} {campaign_suffix} {random.randint(2020, 2024)}"
        
        # Generate campaign type
        campaign_type = random.choice(campaign_types)
        
        # Generate target segment
        target_segment = random.choice(target_segments)
        
        # Generate campaign dates
        start_date = random_date(START_DATE, END_DATE - timedelta(days=30))
        duration_days = random.choice([7, 14, 30, 45, 60, 90])
        end_date = min(start_date + timedelta(days=duration_days), END_DATE)
        
        # Generate budget - varies by campaign type
        if campaign_type in ['TV Advertisement', 'Radio Advertisement']:
            budget = round(random.uniform(50000, 200000), 2)
        elif campaign_type in ['Print Advertisement', 'Influencer Marketing']:
            budget = round(random.uniform(20000, 80000), 2)
        elif campaign_type in ['Social Media', 'Search Engine Marketing', 'Online Display Ads']:
            budget = round(random.uniform(10000, 50000), 2)
        else:
            budget = round(random.uniform(5000, 30000), 2)
        
        # Generate impressions, clicks, conversions with realistic ratios
        base_impressions = random.randint(10000, 1000000)
        
        # Different click-through rates (CTR) by campaign type
        if campaign_type == 'Email Marketing':
            ctr = random.uniform(0.015, 0.04)  # 1.5-4% CTR
        elif campaign_type == 'Search Engine Marketing':
            ctr = random.uniform(0.03, 0.08)  # 3-8% CTR
        elif campaign_type == 'Social Media':
            ctr = random.uniform(0.005, 0.02)  # 0.5-2% CTR
        else:
            ctr = random.uniform(0.001, 0.01)  # 0.1-1% CTR
            
        # Different conversion rates by campaign type
        if campaign_type == 'Email Marketing':
            cvr = random.uniform(0.02, 0.1)  # 2-10% of clicks convert
        elif campaign_type == 'Search Engine Marketing':
            cvr = random.uniform(0.05, 0.15)  # 5-15% of clicks convert
        elif campaign_type == 'In-Store Promotion':
            cvr = random.uniform(0.2, 0.4)  # 20-40% of in-store interactions convert
        else:
            cvr = random.uniform(0.01, 0.05)  # 1-5% of clicks convert
            
        # Calculate metrics
        impressions = int(base_impressions * (budget / 10000))  # Scale impressions by budget
        clicks = int(impressions * ctr)
        conversions = int(clicks * cvr)
        
        # Ensure logical progression
        clicks = min(clicks, impressions)
        conversions = min(conversions, clicks)
        
        # Calculate derived metrics
        conversion_rate = round((conversions / clicks) * 100, 2) if clicks > 0 else 0
        
        # Calculate ROI (Return on Investment)
        # Assuming an average order value based on campaign type
        if 'High-Value' in target_segment:
            avg_order_value = random.uniform(500, 1500)
        else:
            avg_order_value = random.uniform(100, 500)
            
        revenue = conversions * avg_order_value
        roi = round(((revenue - budget) / budget) * 100, 2)
        
        # Create campaign record
        campaign = {
            'campaign_id': campaign_id,
            'campaign_name': campaign_name,
            'campaign_type': campaign_type,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'target_segment': target_segment,
            'budget': budget,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'conversion_rate': conversion_rate,
            'roi': roi
        }
        
        data.append(campaign)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def generate_interactions(customer_ids, customer_df, num_records):
    """
    Generate customer interaction data.
    
    Args:
        customer_ids (list): List of customer IDs
        customer_df (pandas.DataFrame): DataFrame containing customer data
        num_records (int): Number of interaction records to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing interaction data
    """
    print(f"Generating {num_records} customer interaction records...")
    
    channels = ['web', 'mobile app', 'in-store kiosk']
    
    # Interaction types by channel
    interaction_types = {
        'web': ['page_view', 'product_view', 'add_to_cart', 'wishlist_add', 'search', 'checkout', 'purchase', 'review'],
        'mobile app': ['app_open', 'product_view', 'add_to_cart', 'wishlist_add', 'search', 'checkout', 'purchase', 'notification_click'],
        'in-store kiosk': ['session_start', 'product_lookup', 'product_view', 'store_map_view', 'inventory_check', 'checkout']
    }
    
    # Duration ranges by interaction type (in seconds)
    duration_ranges = {
        'page_view': (5, 120),
        'product_view': (10, 180),
        'add_to_cart': (1, 10),
        'wishlist_add': (1, 10),
        'search': (5, 60),
        'checkout': (60, 300),
        'purchase': (60, 300),
        'review': (60, 300),
        'app_open': (1, 5),
        'notification_click': (1, 10),
        'session_start': (1, 10),
        'product_lookup': (10, 120),
        'store_map_view': (10, 60),
        'inventory_check': (10, 60)
    }
    
    # Products and pages for interactions
    products = [
        'iPhone 13', 'Samsung Galaxy S22', 'MacBook Pro', 'Dell XPS 15', 'iPad Pro',
        'Sony Bravia TV', 'Bose Headphones', 'Nintendo Switch', 'Amazon Echo',
        'Refrigerator', 'Coffee Maker', 'Sofa', 'Dining Table', 'Area Rug', 'Cookware Set'
    ]
    
    pages = [
        'home', 'product_listing', 'category_smartphones', 'category_laptops', 'category_tvs',
        'category_audio', 'category_gaming', 'category_smart_home', 'category_kitchen',
        'category_furniture', 'search_results', 'checkout', 'account', 'order_history',
        'wishlist', 'support', 'store_locator', 'about_us', 'blog'
    ]
    
    data = []
    
    # Sample customer indices based on activity patterns
    customer_weights = []
    
    for idx, customer_id in enumerate(customer_ids):
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        
        # Handle potential NaN values in age
        age = customer['age'] if not pd.isna(customer['age']) else 35  # Use average age if missing
        preferred_channel = customer['preferred_channel']
        registration_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        
        # Calculate weight - newer and younger customers tend to have more interactions
        time_factor = 1.0 - min((END_DATE - registration_date).days / (5*365), 0.8)  # Favor newer customers
        age_factor = 1.0 - min(age / 80, 0.7)  # Favor younger customers
        
        # Channel factor - online and both channels tend to have more digital interactions
        if preferred_channel == 'online':
            channel_factor = 1.5
        elif preferred_channel == 'both':
            channel_factor = 1.2
        else:
            channel_factor = 0.8
            
        weight = time_factor * age_factor * channel_factor
        customer_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(customer_weights)
    customer_weights = [w/total_weight for w in customer_weights]
    
    # Sample customer indices based on weights
    customer_weights = np.array(customer_weights)
    
    # Handle any NaN or zero values by replacing with small positive values
    customer_weights = np.nan_to_num(customer_weights, nan=1.0/len(customer_weights))
    
    # Ensure weights sum to 1.0
    customer_weights = customer_weights / np.sum(customer_weights)
    
    customer_indices = np.random.choice(
        range(len(customer_ids)), 
        size=num_records, 
        p=customer_weights,
        replace=True
    )
    
    # Generate session IDs for grouping related interactions
    num_sessions = num_records // 5  # Average 5 interactions per session
    session_ids = [str(uuid.uuid4()) for _ in range(num_sessions)]
    
    for i in range(num_records):
        # Generate interaction ID
        interaction_id = str(uuid.uuid4())
        
        # Get the customer for this interaction
        customer_idx = customer_indices[i % len(customer_indices)]
        customer_id = customer_ids[customer_idx]
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        
        # Get customer details
        age = customer['age']
        preferred_channel = customer['preferred_channel']
        registration_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        
        # Determine channel based on preferred channel
        if preferred_channel == 'online':
            channel_weights = [0.7, 0.3, 0.0]  # Mostly web and mobile
        elif preferred_channel == 'in-store':
            channel_weights = [0.3, 0.1, 0.6]  # Mostly in-store
        else:  # both
            channel_weights = [0.4, 0.3, 0.3]  # Mixed
            
        channel = np.random.choice(channels, p=channel_weights)
        
        # Determine interaction type based on channel
        if channel == 'in-store kiosk' and preferred_channel == 'in-store':
            # Higher chance of purchase for in-store customers using kiosks
            interaction_type_weights = [0.15, 0.15, 0.15, 0.15, 0.2, 0.2]  # Emphasis on checkout
            # Normalize weights to ensure they sum to 1.0
            interaction_type_weights = [w/sum(interaction_type_weights) for w in interaction_type_weights]
            interaction_type = np.random.choice(interaction_types[channel], p=interaction_type_weights)
        else:
            # Random selection with uniform distribution
            interaction_type = np.random.choice(interaction_types[channel])
        
        # Determine interaction date (after registration)
        days_since_registration = (END_DATE - registration_date).days
        if days_since_registration <= 1:
            # For very recent registrations, set interaction on same day
            interaction_date = registration_date
        else:
            # Otherwise, generate a random date after registration
            max_days = min(days_since_registration, 365 * 5)  # Cap at 5 years or days since registration
            random_days = np.random.randint(0, max_days)
            interaction_date = registration_date + timedelta(days=random_days)
            
        # Apply time of day variation
        hour = np.random.normal(14, 5)  # Peak around 2 PM
        hour = max(min(int(hour), 23), 0)  # Bound between 0-23
        minute = random.randint(0, 59)
        interaction_date = interaction_date.replace(hour=hour, minute=minute)
        
        # Determine duration based on interaction type
        min_duration, max_duration = duration_ranges.get(interaction_type, (5, 60))
        duration = random.randint(min_duration, max_duration)
        
        # Determine page or product
        if interaction_type in ['product_view', 'add_to_cart', 'wishlist_add', 'product_lookup', 'inventory_check']:
            page_or_product = random.choice(products)
        else:
            page_or_product = random.choice(pages)
            
        # Assign to session
        session_id = session_ids[i % num_sessions]
        
        # Create interaction record
        interaction = {
            'interaction_id': interaction_id,
            'customer_id': customer_id,
            'channel': channel,
            'interaction_type': interaction_type,
            'interaction_date': interaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration,
            'page_or_product': page_or_product,
            'session_id': session_id
        }
        
        data.append(interaction)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by customer and date to group sessions more realistically
    df = df.sort_values(['customer_id', 'interaction_date'])
    
    # Reassign session IDs to have sequential interactions in the same session
    customer_session_counter = {}
    new_session_ids = []
    
    for idx, row in df.iterrows():
        customer_id = row['customer_id']
        interaction_date = datetime.strptime(row['interaction_date'], '%Y-%m-%d %H:%M:%S')
        
        # Check if this is a new customer or if too much time has passed since last interaction
        if customer_id not in customer_session_counter:
            customer_session_counter[customer_id] = {
                'last_date': interaction_date,
                'session_count': 0
            }
            new_session_id = f"{customer_id}_session_{customer_session_counter[customer_id]['session_count']}"
        else:
            # If more than 30 minutes since last interaction, create new session
            time_diff = (interaction_date - customer_session_counter[customer_id]['last_date']).total_seconds() / 60
            if time_diff > 30:
                customer_session_counter[customer_id]['session_count'] += 1
            
            customer_session_counter[customer_id]['last_date'] = interaction_date
            new_session_id = f"{customer_id}_session_{customer_session_counter[customer_id]['session_count']}"
        
        new_session_ids.append(new_session_id)
    
    df['session_id'] = new_session_ids
    
    return df

def generate_support_tickets(customer_ids, customer_df, num_records):
    """
    Generate customer support ticket data.
    
    Args:
        customer_ids (list): List of customer IDs
        customer_df (pandas.DataFrame): DataFrame containing customer data
        num_records (int): Number of support ticket records to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing support ticket data
    """
    print(f"Generating {num_records} support ticket records...")
    
    issue_categories = ['Billing', 'Technical', 'Product Inquiry', 'Returns', 'Shipping', 'Website Issue', 'Account Issue']
    priorities = ['Low', 'Medium', 'High']
    statuses = ['Resolved', 'Pending', 'Escalated', 'Closed without Resolution']
    
    # Issue notes templates
    issue_notes_templates = {
        'Billing': [
            "Customer reported incorrect charge on their {payment_method}.",
            "Customer was charged twice for the same {product}.",
            "Customer couldn't apply discount code at checkout.",
            "Customer inquiring about refund status for returned {product}.",
            "Billing address verification failed during checkout."
        ],
        'Technical': [
            "{product} not powering on after purchase.",
            "Customer experiencing software issues with {product}.",
            "Setup assistance needed for {product}.",
            "Customer reporting connectivity issues with {product}.",
            "{product} displaying error code during operation."
        ],
        'Product Inquiry': [
            "Customer asking about {product} specifications.",
            "Customer inquiring about {product} compatibility with their existing setup.",
            "Customer requesting recommendations for products similar to {product}.",
            "Customer asking about warranty coverage for {product}.",
            "Stock availability inquiry for {product}."
        ],
        'Returns': [
            "Customer requesting return authorization for {product}.",
            "Customer received damaged {product} and wants to return it.",
            "Customer experiencing buyer's remorse for {product}.",
            "Customer requesting return policy extension for {product}.",
            "Customer wants to exchange {product} for a different model."
        ],
        'Shipping': [
            "Customer inquiring about delayed shipment of {product}.",
            "Package containing {product} was delivered to wrong address.",
            "Customer requesting expedited shipping for {product}.",
            "Tracking information shows {product} delivery but customer didn't receive it.",
            "Customer requesting address change for shipment of {product}."
        ],
        'Website Issue': [
            "Customer unable to complete checkout process on website.",
            "Customer reporting website error when browsing {product} category.",
            "Customer account login issues on the website.",
            "Images not loading on {product} page.",
            "Customer unable to reset password on website."
        ],
        'Account Issue': [
            "Customer unable to update payment information in account.",
            "Customer requesting deletion of their account.",
            "Customer reporting unauthorized access to their account.",
            "Customer unable to view order history in account.",
            "Customer requesting to merge duplicate accounts."
        ]
    }
    
    # Products for templates
    products = [
        'iPhone', 'Samsung Galaxy', 'MacBook', 'Dell Laptop', 'iPad',
        'Sony TV', 'Bose Headphones', 'Nintendo Switch', 'Amazon Echo',
        'Refrigerator', 'Coffee Maker', 'Sofa', 'Dining Table', 'Area Rug', 'Cookware Set'
    ]
    
    payment_methods = ['credit card', 'debit card', 'PayPal', 'Apple Pay', 'gift card']
    
    data = []
    
    # Customer weights for support tickets
    # Some customers are more likely to submit support tickets than others
    customer_weights = []
    
    for idx, customer_id in enumerate(customer_ids):
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        
        # Handle potential NaN values in age
        age = customer['age'] if not pd.isna(customer['age']) else 35  # Use average age if missing
        
        # Older customers and very young customers tend to submit more support tickets
        if age > 60 or age < 25:
            age_factor = 1.5
        else:
            age_factor = 1.0
            
        # Simple random factor to simulate some customers being more prone to issues
        random_factor = random.uniform(0.5, 2.0)
        
        weight = age_factor * random_factor
        customer_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(customer_weights)
    customer_weights = [w/total_weight for w in customer_weights]
    
    # Sample customer indices based on weights
    customer_indices = np.random.choice(
        range(len(customer_ids)), 
        size=num_records, 
        p=customer_weights,
        replace=True
    )
    
    for i in range(num_records):
        # Generate ticket ID
        ticket_id = str(uuid.uuid4())
        
        # Get the customer for this ticket
        customer_idx = customer_indices[i]
        customer_id = customer_ids[customer_idx]
        customer = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
        
        # Get customer details
        registration_date = datetime.strptime(customer['registration_date'], '%Y-%m-%d')
        
        # Select issue category
        issue_category = random.choice(issue_categories)
        
        # Set priority based on issue category
        if issue_category in ['Technical', 'Billing']:
            priority_weights = [0.1, 0.4, 0.5]  # Higher chance of High priority
        elif issue_category in ['Returns', 'Shipping']:
            priority_weights = [0.2, 0.5, 0.3]  # Medium priority most common
        else:
            priority_weights = [0.5, 0.4, 0.1]  # Mostly Low priority
            
        priority = np.random.choice(priorities, p=priority_weights)
        
        # Determine submission date (after registration)
        days_since_registration = (END_DATE - registration_date).days
        if days_since_registration <= 1:
            # For very recent registrations, set submission on same day
            submission_date = registration_date
        else:
            # Otherwise, generate a random date after registration
            max_days = min(days_since_registration, 365 * 5)  # Cap at 5 years or days since registration
            random_days = np.random.randint(0, max_days)
            submission_date = registration_date + timedelta(days=random_days)
        
        # Determine resolution status and date
        if random.random() < 0.05:  # 5% still pending
            status = 'Pending'
            resolution_date = None
            resolution_time_hours = None
        elif random.random() < 0.03:  # 3% escalated
            status = 'Escalated'
            resolution_date = None
            resolution_time_hours = None
        elif random.random() < 0.02:  # 2% closed without resolution
            status = 'Closed without Resolution'
            
            # Resolution date 1-10 days after submission
            resolution_days = random.randint(1, 10)
            resolution_date = submission_date + timedelta(days=resolution_days)
            resolution_date = min(resolution_date, END_DATE)  # Ensure it's not in the future
            
            # Calculate resolution time in hours
            resolution_time_hours = resolution_days * 24
        else:  # 90% resolved
            status = 'Resolved'
            
            # Resolution time based on priority
            if priority == 'High':
                # High priority tickets resolved in 1-48 hours
                resolution_hours = random.randint(1, 48)
            elif priority == 'Medium':
                # Medium priority tickets resolved in 4-72 hours
                resolution_hours = random.randint(4, 72)
            else:  # Low
                # Low priority tickets resolved in 24-120 hours
                resolution_hours = random.randint(24, 120)
                
            resolution_date = submission_date + timedelta(hours=resolution_hours)
            resolution_date = min(resolution_date, END_DATE)  # Ensure it's not in the future
            resolution_time_hours = resolution_hours
        
        # Generate customer satisfaction score
        if status == 'Resolved':
            if resolution_time_hours is not None:
                # Satisfaction depends on resolution time and priority
                if priority == 'High' and resolution_time_hours <= 24:
                    satisfaction_weights = [0.05, 0.1, 0.15, 0.3, 0.4]  # Higher chance of 5
                elif priority == 'High' and resolution_time_hours > 24:
                    satisfaction_weights = [0.1, 0.2, 0.3, 0.25, 0.15]  # Lower for slow high-priority
                elif priority == 'Medium' and resolution_time_hours <= 48:
                    satisfaction_weights = [0.05, 0.1, 0.2, 0.35, 0.3]  # Good for medium in good time
                else:
                    satisfaction_weights = [0.1, 0.15, 0.35, 0.25, 0.15]  # Average
                    
                satisfaction_score = np.random.choice(range(1, 6), p=satisfaction_weights)
            else:
                satisfaction_score = np.random.choice(range(1, 6))
        elif status == 'Closed without Resolution':
            # Lower satisfaction for unresolved issues
            satisfaction_weights = [0.4, 0.3, 0.2, 0.07, 0.03]
            satisfaction_score = np.random.choice(range(1, 6), p=satisfaction_weights)
        else:
            satisfaction_score = None  # No score for pending/escalated
        
        # Generate issue notes using templates
        product = random.choice(products)
        payment_method = random.choice(payment_methods)
        
        template = random.choice(issue_notes_templates[issue_category])
        notes = template.format(product=product, payment_method=payment_method)
        
        # Create support ticket record
        ticket = {
            'ticket_id': ticket_id,
            'customer_id': customer_id,
            'issue_category': issue_category,
            'priority': priority,
            'submission_date': submission_date.strftime('%Y-%m-%d %H:%M:%S'),
            'resolution_date': resolution_date.strftime('%Y-%m-%d %H:%M:%S') if resolution_date else None,
            'resolution_status': status,
            'resolution_time_hours': resolution_time_hours,
            'customer_satisfaction_score': satisfaction_score,
            'notes': notes
        }
        
        data.append(ticket)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values (approximately 2%)
    mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
    df.loc[mask, 'customer_satisfaction_score'] = np.nan
    
    return df

def save_df_to_csv(df, filename, add_missing_values=True):
    """
    Save DataFrame to CSV with optional missing values.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        filename (str): Output filename
        add_missing_values (bool): Whether to add missing values to the data
    """
    if add_missing_values:
        # For each column, add ~2% missing values (except ID columns and dates)
        for column in df.columns:
            if not column.endswith('_id') and not column.endswith('_date'):
                mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
                df.loc[mask, column] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")

def main():
    """Main function to generate all datasets."""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    print(f"Fusion Retail Synthetic Data Generator")
    print(f"--------------------------------------")
    
    # Generate customer data
    customers_df, customer_ids = generate_customers(NUM_CUSTOMERS)
    save_df_to_csv(customers_df, 'output/customers.csv')
    
    # Generate transaction data
    transactions_df = generate_transactions(customer_ids, customers_df, NUM_TRANSACTIONS)
    save_df_to_csv(transactions_df, 'output/transactions.csv')
    
    # Generate campaign data
    campaigns_df = generate_campaigns(NUM_CAMPAIGNS)
    save_df_to_csv(campaigns_df, 'output/campaigns.csv')
    
    # Generate interaction data
    interactions_df = generate_interactions(customer_ids, customers_df, NUM_INTERACTIONS)
    save_df_to_csv(interactions_df, 'output/interactions.csv')
    
    # Generate support ticket data
    support_tickets_df = generate_support_tickets(customer_ids, customers_df, NUM_SUPPORT_TICKETS)
    save_df_to_csv(support_tickets_df, 'output/support_tickets.csv')
    
    print("\nData generation complete! Files are saved in the 'output' directory.")
    print("\nREADME:")
    print("""
    Fusion Retail Synthetic Data Generator
    --------------------------------------
    
    This program has generated synthetic data for Fusion Retail, a fictitious omnichannel retailer
    specializing in consumer electronics and home goods. The data includes customer profiles,
    transactions, marketing campaigns, customer interactions, and support tickets.
    
    Files Generated:
    1. customers.csv - 5,000 customer records with demographic information
    2. transactions.csv - 50,000 transaction records with product and purchase details
    3. campaigns.csv - 200 marketing campaign records with performance metrics
    4. interactions.csv - 100,000 customer interaction records across various channels
    5. support_tickets.csv - 3,000 customer support ticket records
    
    Data Model:
    - Each customer has a unique customer_id that links to their transactions, interactions, and support tickets
    - Transaction dates are always after customer registration dates
    - Interactions follow realistic patterns based on customer demographics and preferences
    - Support tickets include realistic resolution times and satisfaction scores
    - Marketing campaigns include calculated metrics like conversion rate and ROI
    
    Data Quality:
    - Approximately 2% of values are missing to simulate real-world data issues
    - Some duplicate transactions have been added
    - Price and quantity outliers are included
    - Realistic distributions for customer demographics, product preferences, and sales patterns
    
    Usage:
    The generated CSV files can be used for data analysis, visualization, and machine learning projects.
    They provide a comprehensive dataset for exploring retail analytics scenarios including:
    - Customer segmentation
    - Sales analysis
    - Marketing campaign effectiveness
    - Customer journey analysis
    - Support ticket resolution performance
    
    To regenerate the data with different random values, run the script again with a different seed:
    python fusion_retail_data_generator.py --seed 123
    
    To adjust the volume of data, use the scale parameter:
    python fusion_retail_data_generator.py --scale 0.5  # Half the default volume
    python fusion_retail_data_generator.py --scale 2.0  # Double the default volume
    """)

if __name__ == "__main__":
    main()