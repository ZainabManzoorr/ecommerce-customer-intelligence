import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate ecommerce dataset.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin1")

    expected_cols = ["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]
    missing = [col for col in expected_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    logger.info(f"Dataset loaded with shape: {df.shape}")

    return df