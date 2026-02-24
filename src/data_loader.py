"""
Load and join Olist CSV tables into a clean modeling dataframe.
Validate row counts, handle missing values.
Expects data from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# Default CSV filenames (Kaggle Olist dataset)
DEFAULT_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_raw_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all Olist CSVs from data_dir. Returns dict of table_name -> DataFrame."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    tables = {}
    for key, filename in DEFAULT_FILES.items():
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Expected CSV not found: {filepath}")
        tables[key] = pd.read_csv(filepath)
    return tables


def _parse_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_orders_with_target(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join orders + reviews to get one row per order with target.
    target = 1 if review_score <= 2 else 0.
    Drops orders with missing review_score.
    """
    orders = tables["orders"].copy()
    reviews = tables["reviews"].copy()

    # One review per order (take latest if multiple)
    review_cols = ["order_id", "review_score"]
    if "review_creation_date" in reviews.columns:
        reviews = reviews.sort_values("review_creation_date").drop_duplicates(
            subset=["order_id"], keep="last"
        )
    reviews = reviews[review_cols]

    orders = orders.merge(reviews, on="order_id", how="inner")
    orders["target"] = (orders["review_score"] <= 2).astype(int)
    return orders


def build_modeling_dataframe(
    data_dir: str | Path,
    validate: bool = True,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load all CSVs, join relational tables, validate, handle missing values,
    and return a clean modeling dataframe (one row per order with target).

    Args:
        data_dir: Directory containing the Olist CSV files.
        validate: If True, print row counts and basic validation.
        save_path: If set, save the final dataframe to this path (e.g. data/modeling.parquet).

    Returns:
        DataFrame with one row per order, target, and joined attributes for feature engineering.
    """
    data_path = Path(data_dir)
    tables = load_raw_tables(data_path)

    if validate:
        print("Raw table row counts:")
        for name, df in tables.items():
            print(f"  {name}: {len(df):,}")

    # Base: orders with target (only orders that have a review)
    orders = build_orders_with_target(tables)
    if validate:
        print(f"\nOrders with review (target defined): {len(orders):,}")

    # Parse order timestamps for time-based split
    orders = _parse_dates(
        orders,
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    # Order-level aggregates from order_items (order_value, freight_value, num_items, product/seller)
    items = tables["order_items"].copy()
    items = _parse_dates(items, ["shipping_limit_date"])
    order_agg = items.groupby("order_id").agg(
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        num_items=("order_item_id", "count"),
        # First product and seller for order-level (feature eng can refine)
        product_id=("product_id", "first"),
        seller_id=("seller_id", "first"),
    ).reset_index()
    orders = orders.merge(order_agg, on="order_id", how="left")

    # Fill missing order_value/freight from items (should be rare if join is left from orders with review)
    orders["order_value"] = orders["order_value"].fillna(0)
    orders["freight_value"] = orders["freight_value"].fillna(0)
    orders["num_items"] = orders["num_items"].fillna(0).astype(int)
    orders["product_id"] = orders["product_id"].fillna("")
    orders["seller_id"] = orders["seller_id"].fillna("")

    # Payment: primary payment per order (first row per order_id)
    payments = tables["order_payments"].copy()
    payments = payments.sort_values("payment_sequential").drop_duplicates(
        subset=["order_id"], keep="first"
    )
    pay_cols = ["order_id", "payment_type", "payment_installments", "payment_value"]
    pay_cols = [c for c in pay_cols if c in payments.columns]
    orders = orders.merge(payments[pay_cols], on="order_id", how="left")
    orders["payment_type"] = orders["payment_type"].fillna("unknown")
    orders["payment_installments"] = orders["payment_installments"].fillna(0).astype(int)

    # Customers
    customers = tables["customers"].copy()
    cust_cols = ["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]
    cust_cols = [c for c in cust_cols if c in customers.columns]
    orders = orders.merge(customers[cust_cols], on="customer_id", how="left")

    # Sellers (we have seller_id from order_agg)
    sellers = tables["sellers"].copy()
    seller_cols = ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
    seller_cols = [c for c in seller_cols if c in sellers.columns]
    orders = orders.merge(sellers[seller_cols], on="seller_id", how="left")

    # Products + category translation
    products = tables["products"].copy()
    prod_cols = [
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    prod_cols = [c for c in prod_cols if c in products.columns]
    orders = orders.merge(products[prod_cols], on="product_id", how="left")

    trans = tables.get("category_translation")
    if trans is not None and "product_category_name" in trans.columns:
        en_col = "product_category_name_english" if "product_category_name_english" in trans.columns else trans.columns[1]
        orders = orders.merge(
            trans[["product_category_name", en_col]].rename(columns={en_col: "product_category_english"}),
            on="product_category_name",
            how="left",
        )
        orders["product_category_english"] = orders["product_category_english"].fillna(orders["product_category_name"].astype(str))
    else:
        orders["product_category_english"] = orders["product_category_name"].fillna("").astype(str)

    # Geolocation: customer and seller lat/lng (aggregate by zip prefix)
    geo = tables["geolocation"].copy()
    if "geolocation_zip_code_prefix" in geo.columns:
        geo_agg = geo.groupby("geolocation_zip_code_prefix").agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
        ).reset_index()
        geo_agg = geo_agg.rename(columns={"geolocation_zip_code_prefix": "zip_prefix"})
        orders["customer_zip_code_prefix"] = orders["customer_zip_code_prefix"].fillna(0).astype(int)
        orders["seller_zip_code_prefix"] = orders["seller_zip_code_prefix"].fillna(0).astype(int)
        cust_geo = geo_agg.rename(columns={"zip_prefix": "customer_zip_code_prefix", "lat": "customer_lat", "lng": "customer_lng"})
        orders = orders.merge(cust_geo, on="customer_zip_code_prefix", how="left")
        sell_geo = geo_agg.rename(columns={"zip_prefix": "seller_zip_code_prefix", "lat": "seller_lat", "lng": "seller_lng"})
        orders = orders.merge(sell_geo, on="seller_zip_code_prefix", how="left")

    # Drop duplicate columns from merges if any
    orders = orders.loc[:, ~orders.columns.duplicated()]

    if validate:
        print(f"\nFinal modeling dataframe: {len(orders):,} rows, {len(orders.columns)} columns")
        print(f"Target distribution: {orders['target'].value_counts().to_dict()}")
        print(f"Missing key cols: {orders.isna().sum()[orders.isna().sum() > 0].to_dict()}")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            orders.to_parquet(out, index=False)
        else:
            orders.to_csv(out, index=False)
        print(f"Saved to {out}")

    return orders


def load_modeling_data(path: str | Path) -> pd.DataFrame:
    """Load a previously saved modeling dataframe from parquet or CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling data not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    import os

    data_dir = os.environ.get("OLIST_DATA_DIR", "data")
    save_path = os.environ.get("OLIST_MODELING_PATH", "data/modeling.parquet")
    print(f"Data dir: {data_dir}")
    print(f"Save path: {save_path}")
    df = build_modeling_dataframe(data_dir, validate=True, save_path=save_path)
    print("Done.")
