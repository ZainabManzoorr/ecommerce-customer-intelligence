from src.data_ingestion.load_data import load_dataset
from src.preprocessing.clean_data import DataCleaner

df = load_dataset("data/raw/Transactions.csv")

cleaner = DataCleaner(df)

clean_df = (
    cleaner
    .drop_missing_customers()
    .remove_duplicates()
    .remove_invalid_transactions()
    .fix_revenue()
    .clean_datetime()
    .remove_outliers()
    .get_clean_data()
)

print("\nDATA AUDIT REPORT")
print(cleaner.data_audit())