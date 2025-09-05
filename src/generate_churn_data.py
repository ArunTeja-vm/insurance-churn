# Part 1: Imports, Helpers, Vehicle Functions
import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

faker = Faker('en_US')

state_abbr_map = {
    'California': 'CA', 'Texas': 'TX', 'Florida': 'FL',
    'New York': 'NY', 'Illinois': 'IL', 'Pennsylvania': 'PA'
}

# Occupation assignment by age
def assign_occupation(age):
    if age < 25:
        return 'Student'
    elif age < 60:
        return random.choices(
            ['IT', 'Finance', 'Govt', 'Self-employed'],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]
    else:
        return 'Retired'

# Income assignment by occupation
def assign_income(occupation):
    income_ranges = {
        'Student': (5000, 15000),
        'IT': (60000, 150000),
        'Finance': (70000, 200000),
        'Govt': (40000, 90000),
        'Self-employed': (30000, 250000),
        'Retired': (20000, 60000)
    }
    low, high = income_ranges.get(occupation, (30000, 100000))
    return round(np.random.uniform(low, high), -3)

# Vehicle details generator
def get_make_model_year_age_value():
    make_model_base = {
        'Toyota': {'Camry': 28000, 'Corolla': 22000, 'RAV4': 30000},
        'Ford': {'F-150': 35000, 'Focus': 20000, 'Escape': 27000},
        'Honda': {'Civic': 23000, 'Accord': 27000, 'CR-V': 29000},
        'Chevrolet': {'Impala': 31000, 'Malibu': 24000, 'Equinox': 26000},
        'Nissan': {'Altima': 25000, 'Sentra': 20000, 'Rogue': 27000}
    }
    current_year = datetime.today().year
    make = random.choice(list(make_model_base.keys()))
    model = random.choice(list(make_model_base[make].keys()))
    base_price = make_model_base[make][model]
    model_year = random.randint(2010, current_year)
    age = current_year - model_year
    if age == 0:
        depreciation = 0.12
    else:
        depreciation = min(0.12 + 0.08 * (age - 1), 0.8)
    value = max(round(base_price * (1 - depreciation), 2), 2000)
    if value >= 15000:
        coverage = 'Medical'
    elif age <= 10 and value >= 10000:
        coverage = 'Collision'
    else:
        coverage = 'Liability'
    return {
        "Vehicle_Make": make,
        "Vehicle_Model": model,
        "Model_Year": model_year,
        "Vehicle_Age": age,
        "Vehicle_Value": value,
        "Coverage": coverage
    }

def generate_churn_data(n=25000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    today = datetime.today()

    us_states = ['California', 'Texas', 'Florida', 'New York', 'Illinois', 'Pennsylvania']
    genders = ['Male', 'Female', 'Other']
    marital_statuses = ['Single', 'Married', 'Divorced']
    billing_methods = ['Auto-Pay', 'Manual', 'EFT']
    billing_frequencies = ['Monthly', 'Quarterly', 'Annually']
    payment_methods = ['Card', 'ACH']
    safety_features_list = ['ABS', 'Airbags', 'Lane Assist', 'Backup Camera', 'Blind Spot Monitor']
    claim_types = ['Collision', 'Weather', 'Theft']
    claim_outcomes = ['Approved', 'Declined']
    reasons_for_cancellation = ['Price', 'Service', 'Switched', 'Moved']
    reason_codes = ['P01', 'S01', 'C01', 'M01']

    def random_date(start_days_ago=365*10, end_days_ago=30):
        delta = random.randint(end_days_ago, start_days_ago)
        return today - timedelta(days=delta)

    data = []
    for _ in range(n):
        state = np.random.choice(us_states)
        vehicle = get_make_model_year_age_value()
        age = np.random.randint(18, 65)
        occupation = assign_occupation(age)
        income = assign_income(occupation)
        policy_start = random_date(start_days_ago=365*4, end_days_ago=0)
        tenure = (today - policy_start).days // 30
        policy_end = policy_start + timedelta(days=365)
        premium_change_pct = round(min(max(np.random.normal(loc=7, scale=5), 0), 25), 2) if tenure >= 12 else 0
        retention_offer_sent = int(random.random() < 0.3)
        retention_offer_accepted = int(retention_offer_sent and random.random() < 0.5)
        lifetime_claims = np.random.randint(0, 6) if tenure >= 12 else 0
        claims_last_3_years = np.random.randint(0, min(lifetime_claims + 1, 4))
        at_fault_accidents = np.random.randint(0, min(claims_last_3_years + 1, 3))
        has_claim = lifetime_claims > 0
        claim_outcome = random.choice(claim_outcomes) if has_claim else None
        claim_type = random.choice(claim_types) if has_claim else None
        resolution_days = np.random.randint(1, 90) if has_claim else None
        claim_payout = 0 if (not has_claim or claim_outcome == "Declined") else round(np.random.uniform(500, 15000), 2)
        csat_score = np.random.randint(60, 101) if not has_claim else (
            np.random.randint(5, 21) if claim_outcome == "Declined" else (
                np.random.randint(20, 51) if resolution_days > 45 else np.random.randint(50, 81)
            )
        )
        sentiment_score = (
            round(np.random.uniform(0.5, 1.0), 2)
            if lifetime_claims == 0 else (
                round(np.random.uniform(-0.2, 0.5), 2) if csat_score < 50 else round(np.random.uniform(0.3, 0.9), 2)
            )
        )
        interaction_score = round(np.random.uniform(0, 1), 2)
        nps = (
            np.random.randint(9, 11) if csat_score >= 80 else (
                np.random.randint(6, 9) if csat_score >= 50 else np.random.randint(0, 6)
            )
        )
        complaint_count = np.random.randint(0, 5)
        payment_method = random.choice(payment_methods)
        vin_validated = 1 if vehicle['Vehicle_Age'] < 10 and vehicle['Vehicle_Value'] > 8000 else np.random.choice([0,1], p=[0.3,0.7])
        claim_closed_date = today - timedelta(days=random.randint(1,300)) if has_claim else None

        # Churn scoring logic
        risk_points = 0
        if premium_change_pct > 12:
            risk_points += 3
        if tenure < 12:
            risk_points += 2
        if complaint_count >= 2:
            risk_points += 2
        if sentiment_score < 0:
            risk_points += 2
        if csat_score < 40:
            risk_points += 2
        if retention_offer_accepted == 0:
            risk_points += 1
        if nps >= 9 and csat_score >= 80:
            risk_points -= 3
        if lifetime_claims > 0 and csat_score <= 30 and resolution_days and resolution_days > 45:
            risk_points += 3
        if tenure >= 180 and lifetime_claims == 0 and premium_change_pct > 10:
            risk_points += 2
        churn_prob = (
            0.05 if risk_points <= 1 else
            0.15 if risk_points <= 3 else
            0.35 if risk_points <= 5 else
            0.6 if risk_points <= 7 else
            0.85
        )
        churned = np.random.binomial(1, churn_prob)
        reason_for_cancel = random.choice(reasons_for_cancellation) if churned else 'None'
        reason_code = reason_codes[reasons_for_cancellation.index(reason_for_cancel)] if churned else 'N/A'

        record = {
            "Customer_ID": str(uuid.uuid4()),
            "Customer_Age": age,
            "Gender": random.choice(genders),
            "Marital_Status": random.choice(marital_statuses),
            "State": state,
            "Postal_Code": faker.zipcode_in_state(state_abbr=state_abbr_map[state]),
            "Income": income,
            "Address_Change_Flag": np.random.choice([0, 1], p=[0.9, 0.1]),
            "Vehicle_Age": vehicle["Vehicle_Age"],
            "Vehicle_Make": vehicle["Vehicle_Make"],
            "Vehicle_Model": vehicle["Vehicle_Model"],
            "Vehicle_Value": vehicle["Vehicle_Value"],
            "Annual_Mileage_Estimate": np.random.randint(5000, 20000),
            "Safety_Features": ", ".join(random.sample(safety_features_list, k=2)),
            "VIN_Validated": vin_validated,
            "Policy_Number": str(uuid.uuid4()),
            "Policy_Status": 'Cancelled' if churned else 'Active',
            "Policy_Effective_Date": policy_start.date(),
            "Policy_Expiry_Date": policy_end.date(),
            "Policy_Cancellation_Date": today.date() if churned else None,
            "Policy_Tenure_Months": tenure,
            "Coverage_Type": vehicle["Coverage"],
            "Deductibles": random.choice([250,500,1000]),
            "Number_of_Drivers": np.random.randint(1,3),
            "Number_of_Vehicles": np.random.randint(1,3),
            "Has_Multi_Policy": np.random.choice([0,1],p=[0.75,0.25]),
            "Loyalty_Program_Enrollment": np.random.choice([0,1],p=[0.8,0.2]),
            "Billing_Method": random.choice(billing_methods),
            "Billing_Frequency": random.choice(billing_frequencies),
            "Payment_Method": payment_method,
            "Discount_Count": np.random.randint(3,5) if tenure >25 else np.random.randint(0,3),
            "Premium_Amount": round(np.random.uniform(1000,3000),2),
            "Premium_Change_Percent_Last_Renewal": premium_change_pct,
            "Late_Payment_Count": np.random.randint(0,4),
            "Auto_Renew_Enabled": 1 if payment_method=="Auto-Pay" else 0,
           # "Retention_Offer_Sent": retention_offer_sent,
            #"Retention_Offer_Accepted": retention_offer_accepted,
            "Claims_Count_Lifetime": lifetime_claims,
            "Claims_Count_Last_3_Years": claims_last_3_years,
            "At_Fault_Accident_Count": at_fault_accidents,
            "Time_Since_Last_Claim": np.random.randint(0,60) if has_claim else None,
            "Claim_Type": claim_type,
            "Claim_Outcome": claim_outcome,
            "Total_Claim_Payout_Amount": claim_payout,
            "Claim_Closed_Date": claim_closed_date.date() if claim_closed_date else None,
            "Avg_Claim_Resolution_Days": resolution_days,
            "Customer_Satisfaction_Score": csat_score,
            "Interaction_Score": interaction_score,
            "NPS": nps,
            "Complaint_Count": complaint_count,
            "Sentiment_Score": sentiment_score,
            #"Reason_for_Last_Cancellation": reason_for_cancel,
            #"Reason_Code": reason_code,
            "Churned": churned
        }
        data.append(record)
    return pd.DataFrame(data)

# Generate and save
df = generate_churn_data()
df.to_csv("synthetic_auto_retention_full.csv", index=False)
print("✅ Full data generated successfully.")
