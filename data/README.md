# Olist data

Place the Brazilian E-Commerce (Olist) CSV files here after downloading from Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Required files:

- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

Then from the project root run:

```bash
python -m src.data_loader
```

Or with custom paths:

```bash
OLIST_DATA_DIR=/path/to/csvs OLIST_MODELING_PATH=data/modeling.parquet python -m src.data_loader
```

Output: `data/modeling.parquet` (clean modeling dataframe, one row per order with target).
