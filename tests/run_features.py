from src.features.features import FeatureEngineer
from tests.run_data import clean_df

engineer = FeatureEngineer(clean_df)

feature_df = (
    engineer
    .create_rfm()
    .create_basket_features()
    .create_time_features()
    .build_feature_table()
)

print(feature_df.head())