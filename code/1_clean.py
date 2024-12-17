import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from datetime import datetime
from difflib import get_close_matches


# ----------------- Data Loading -----------------
print('Loading data...')

data_path = 'data/original/train.csv'
df = pd.read_csv(data_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=314)
final_train_df = df.copy()

predictive_path = 'data/original/predictive.csv'
predictive_df = pd.read_csv(predictive_path)


# ----------------- Numerical Variables -----------------
print('Processing numerical variables...')

numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_columns:
    numerical_columns.remove('id')

predictive_numerical_columns = numerical_columns.copy()
predictive_numerical_columns.remove('price')
predictive_numerical_columns.append('id')

numerical_train_df = train_df[numerical_columns].copy()
numerical_test_df = test_df[numerical_columns].copy()
numerical_final_train_df = final_train_df[numerical_columns].copy()
numerical_predictive_df = predictive_df[predictive_numerical_columns].copy()


# ----------------- Missing Values -----------------
print('Processing missing values...')

numerical_train_df = numerical_train_df.replace('null', np.nan)
numerical_test_df = numerical_test_df.replace('null', np.nan)
numerical_final_train_df = numerical_final_train_df.replace('null', np.nan)
numerical_predictive_df = numerical_predictive_df.replace('null', np.nan)

mean_train = numerical_train_df.mean()
numerical_train_df = numerical_train_df.fillna(mean_train)
numerical_test_df = numerical_test_df.fillna(mean_train)

mean_final_train = numerical_final_train_df.mean()
numerical_final_train_df = numerical_final_train_df.fillna(mean_final_train)
numerical_predictive_df = numerical_predictive_df.fillna(mean_final_train)


# ----------------- Categorical Variables -----------------
print('Processing categorical variables...')

bool_columns = [
    'host_is_superhost',
    'host_identity_verified',
    'host_has_profile_pic',
    'instant_bookable',
    'has_availability'
]

for df_part in [train_df, test_df, final_train_df, predictive_df]:
    df_part[bool_columns] = df_part[bool_columns].fillna(False).astype(int)

response_time_mapping = {
    'a few days or more': 0,
    'within a day': 1,
    'within a few hours': 2,
    'within an hour': 3
}

for df_part in [train_df, test_df, final_train_df, predictive_df]:
    df_part['host_response_time'] = df_part['host_response_time'].map(response_time_mapping).fillna(0).astype(int)

room_type_mapping = {
    'Shared room': 0,
    'Private room': 1,
    'Hotel room': 2,
    'Entire home/apt': 3
}

for df_part in [train_df, test_df, final_train_df, predictive_df]:
    df_part['room_type'] = df_part['room_type'].map(room_type_mapping).fillna(-1).astype(int)

def extract_verifications(verifications_str):
    if pd.isna(verifications_str):
        return []
    try:
        return ast.literal_eval(verifications_str)
    except:
        return []

for df_part in [train_df, test_df, final_train_df, predictive_df]:
    df_part['host_verifications_list'] = df_part['host_verifications'].apply(extract_verifications)
    df_part['phone'] = df_part['host_verifications_list'].apply(lambda x: 1 if 'phone' in x else 0)
    df_part['work_email'] = df_part['host_verifications_list'].apply(lambda x: 1 if 'work_email' in x else 0)
    df_part['email'] = df_part['host_verifications_list'].apply(lambda x: 1 if 'email' in x else 0)

for df_part in [train_df, test_df, final_train_df, predictive_df]:
    df_part['brooklyn'] = df_part['neighbourhood_group_cleansed'].apply(lambda x: 1 if 'Brooklyn' == x else 0)
    df_part['manhattan'] = df_part['neighbourhood_group_cleansed'].apply(lambda x: 1 if 'Manhattan' == x else 0)
    df_part['queens'] = df_part['neighbourhood_group_cleansed'].apply(lambda x: 1 if 'Queens' == x else 0)
    df_part['bronx'] = df_part['neighbourhood_group_cleansed'].apply(lambda x: 1 if 'Bronx' == x else 0)
    df_part['staten_island'] = df_part['neighbourhood_group_cleansed'].apply(lambda x: 1 if 'Staten Island' == x else 0)

train_df = train_df.drop(['host_verifications', 'host_verifications_list'], axis=1)
test_df = test_df.drop(['host_verifications', 'host_verifications_list'], axis=1)
final_train_df = final_train_df.drop(['host_verifications', 'host_verifications_list'], axis=1)
predictive_df = predictive_df.drop(['host_verifications', 'host_verifications_list'], axis=1)

non_numerical_columns = bool_columns + [
    'host_response_time',
    'room_type',
    'phone',
    'work_email',
    'email',
    'brooklyn',
    'manhattan',
    'queens',
    'bronx',
    'staten_island'
]

non_numerical_train_df = train_df[non_numerical_columns].copy()
non_numerical_test_df = test_df[non_numerical_columns].copy()
non_numerical_final_train_df = final_train_df[non_numerical_columns].copy()
non_numerical_predictive_df = predictive_df[non_numerical_columns].copy()


# ----------------- Dates -----------------
print('Processing dates...')

reference_date = datetime(2024, 11, 24)
date_features = ['host_since_days_since', 'first_review_days_since', 'last_review_days_since', 'no_review']

def process_dates(df, stats=None, is_train=False):
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    df['host_since_days_since'] = (reference_date - df['host_since']).dt.days
    df['first_review_days_since'] = (reference_date - df['first_review']).dt.days
    df['last_review_days_since'] = (reference_date - df['last_review']).dt.days

    if is_train:
        min_host_since = df['host_since_days_since'].min()
        mean_first_review = df['first_review_days_since'].mean()
        mean_last_review = df['last_review_days_since'].mean()
        stats = {
            'min_host_since': min_host_since,
            'mean_first_review': mean_first_review,
            'mean_last_review': mean_last_review
        }

    df['host_since_days_since'] = df['host_since_days_since'].fillna(stats['min_host_since'])
    df['first_review_days_since'] = df['first_review_days_since'].fillna(stats['mean_first_review'])
    df['last_review_days_since'] = df['last_review_days_since'].fillna(stats['mean_last_review'])

    df['no_review'] = ((df['first_review'].isna()) & (df['last_review'].isna())).astype(int)

    df = df.drop(['host_since', 'first_review', 'last_review'], axis=1)
    date_df = df[date_features].copy()

    if is_train:
        return date_df, stats
    return date_df

date_train_df, train_stats = process_dates(train_df, is_train=True)
non_numerical_train_df = pd.concat([non_numerical_train_df, date_train_df], axis=1)

date_test_df = process_dates(test_df, stats=train_stats)
non_numerical_test_df = pd.concat([non_numerical_test_df, date_test_df], axis=1)

date_final_train_df, final_train_stats = process_dates(final_train_df, is_train=True)
non_numerical_final_train_df = pd.concat([non_numerical_final_train_df, date_final_train_df], axis=1)

date_predictive_df = process_dates(predictive_df, stats=final_train_stats)
non_numerical_predictive_df = pd.concat([non_numerical_predictive_df, date_predictive_df], axis=1)


# ----------------- Amenities -----------------
print('Processing amenities...')

amenity_targets = {
    'air_conditioning': ['air conditioning', 'AC'],
    'tv': ['TV', 'television'],
    'streaming_services': ['Apple TV', 'Disney+', 'HBO Max', 'Hulu', 'Netflix', 'Roku', 'Amazon Prime'],
    'refrigerator': ['refrigerator', 'fridge'],
    'microwave': ['microwave'],
    'wifi': ['wifi'],
    'parking': ['parking', 'free parking', 'street parking', 'paid parking', 'driveway parking'],
    'gym': ['gym', 'workout', 'exercise'],
    'water_view': ['ocean view', 'sea view', 'water view', 'river view'],
    'kitchen': ['kitchen', 'oven', 'stove']
}

def process_amenities(df):
    amenities_list = df['amenities'].apply(lambda x: x.replace('[', '').replace(']', '').replace('\"', '').split(', '))
    amenities_df = pd.DataFrame(index=df.index)
    for var_name, target_words in amenity_targets.items():
        def check_amenity(amenities):
            for amenity in amenities:
                match = get_close_matches(amenity.lower(), [w.lower() for w in target_words], n=1, cutoff=0.8)
                if match:
                    return 1
            return 0
        amenities_df[var_name] = amenities_list.apply(check_amenity)
    return amenities_df

amenities_train_df = process_amenities(train_df)
non_numerical_train_df = pd.concat([non_numerical_train_df, amenities_train_df], axis=1)

amenities_test_df = process_amenities(test_df)
non_numerical_test_df = pd.concat([non_numerical_test_df, amenities_test_df], axis=1)

amenities_final_train_df = process_amenities(final_train_df)
non_numerical_final_train_df = pd.concat([non_numerical_final_train_df, amenities_final_train_df], axis=1)

amenities_predictive_df = process_amenities(predictive_df)
non_numerical_predictive_df = pd.concat([non_numerical_predictive_df, amenities_predictive_df], axis=1)


# ----------------- Save Data -----------------
print('Saving data...')

combined_train_df = pd.concat([numerical_train_df, non_numerical_train_df], axis=1)
combined_test_df = pd.concat([numerical_test_df, non_numerical_test_df], axis=1)
combined_final_train_df = pd.concat([numerical_final_train_df, non_numerical_final_train_df], axis=1)
combined_predictive_df = pd.concat([numerical_predictive_df, non_numerical_predictive_df], axis=1)

combined_all = pd.concat([combined_train_df, combined_test_df, combined_final_train_df, combined_predictive_df], axis=0)
features_with_missing = combined_all.columns[combined_all.isnull().any()].tolist()

combined_train_csv_path = 'data/processed/1_clean/train.csv'
combined_test_csv_path = 'data/processed/1_clean/test.csv'
combined_final_train_csv_path = 'data/processed/1_clean/final_train.csv'
combined_predictive_csv_path = 'data/processed/1_clean/predictive.csv'

combined_train_df.to_csv(combined_train_csv_path, index=False)
combined_test_df.to_csv(combined_test_csv_path, index=False)
combined_predictive_df.to_csv(combined_predictive_csv_path, index=False)
combined_final_train_df.to_csv(combined_final_train_csv_path, index=False)