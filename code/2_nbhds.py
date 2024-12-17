import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# ----------------- Data Loading -----------------
print('Loading data...')

data_path = 'data/original/train.csv'
df = pd.read_csv(data_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=314)

final_train_df = df.copy()

predictive_path = 'data/original/predictive.csv'
predictive_df = pd.read_csv(predictive_path)

processed_train_csv_path = 'data/processed/1_clean/train.csv'
processed_test_csv_path = 'data/processed/1_clean/test.csv'
processed_final_train_csv_path = 'data/processed/1_clean/final_train.csv'
processed_predictive_csv_path = 'data/processed/1_clean/predictive.csv'

processed_train_df = pd.read_csv(processed_train_csv_path)
processed_test_df = pd.read_csv(processed_test_csv_path)
processed_final_train_df = pd.read_csv(processed_final_train_csv_path)
processed_predictive_df = pd.read_csv(processed_predictive_csv_path)


# ----------------- Neighbourhoods -----------------
print('Processing neighbourhoods...')

def get_neighbourhood_price(row, nn, price_mapping, train_prices):
    neighbourhood = row['neighbourhood_cleansed']
    if neighbourhood in price_mapping:
        return price_mapping[neighbourhood]
    else:
        coord = np.array([row['latitude'], row['longitude']]).reshape(1, -1)
        distances, indices = nn.kneighbors(coord)
        neighbor_prices = train_prices.iloc[indices[0]]
        return neighbor_prices.mean()


neighbourhood_price_mapping = train_df.groupby('neighbourhood_cleansed')['price'].mean().to_dict()

train_coords = train_df[['latitude', 'longitude']].values
train_neighbourhood_prices = train_df['neighbourhood_cleansed'].map(neighbourhood_price_mapping)

train_nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_coords)


final_neighbourhood_price_mapping = final_train_df.groupby('neighbourhood_cleansed')['price'].mean().to_dict()

final_train_coords = final_train_df[['latitude', 'longitude']].values
final_train_neighbourhood_prices = final_train_df['neighbourhood_cleansed'].map(final_neighbourhood_price_mapping)

final_train_nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(final_train_coords)


neighbourhood_train = pd.DataFrame({
    'neighbourhood_price': train_df['neighbourhood_cleansed'].map(neighbourhood_price_mapping)
})
processed_train_df.reset_index(drop=True, inplace=True)
neighbourhood_train.reset_index(drop=True, inplace=True)
processed_train_df = pd.concat([processed_train_df, neighbourhood_train], axis=1)

neighbourhood_test = pd.DataFrame({
    'neighbourhood_price': test_df.apply(get_neighbourhood_price, axis=1, nn=train_nn, price_mapping=neighbourhood_price_mapping, train_prices=train_neighbourhood_prices)
})
processed_test_df.reset_index(drop=True, inplace=True)
neighbourhood_test.reset_index(drop=True, inplace=True)
processed_test_df = pd.concat([processed_test_df, neighbourhood_test], axis=1)

neighbourhood_final_train = pd.DataFrame({
    'neighbourhood_price': final_train_df['neighbourhood_cleansed'].map(final_neighbourhood_price_mapping)
})
processed_final_train_df.reset_index(drop=True, inplace=True)
neighbourhood_final_train.reset_index(drop=True, inplace=True)
processed_final_train_df = pd.concat([processed_final_train_df, neighbourhood_final_train], axis=1)

neighbourhood_predictive = pd.DataFrame({
    'neighbourhood_price': predictive_df.apply(get_neighbourhood_price, axis=1, nn=final_train_nn, price_mapping=final_neighbourhood_price_mapping, train_prices=final_train_neighbourhood_prices)
})
processed_predictive_df.reset_index(drop=True, inplace=True)
neighbourhood_predictive.reset_index(drop=True, inplace=True)
processed_predictive_df = pd.concat([processed_predictive_df, neighbourhood_predictive], axis=1)


# ----------------- Save Data -----------------
print('Saving data...')

processed_train_csv_path = 'data/processed/2_nbhds/train.csv'
processed_test_csv_path = 'data/processed/2_nbhds/test.csv'
processed_final_train_csv_path = 'data/processed/2_nbhds/final_train.csv'
processed_predictive_csv_path = 'data/processed/2_nbhds/predictive.csv'

processed_train_df.to_csv(processed_train_csv_path, index=False)
processed_test_df.to_csv(processed_test_csv_path, index=False)
processed_final_train_df.to_csv(processed_final_train_csv_path, index=False)
processed_predictive_df.to_csv(processed_predictive_csv_path, index=False)