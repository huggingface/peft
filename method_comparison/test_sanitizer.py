import pandas as pd
import pytest

from .sanitizer import parse_and_filter


@pytest.fixture
def df_products():
    data = {
        'product_id': [101, 102, 103, 104, 105, 106],
        'category': ['Electronics', 'Books', 'Electronics', 'Home Goods', 'Books', 'Electronics'],
        'price': [799.99, 19.99, 49.50, 120.00, 24.99, 150.00],
        'stock': [15, 300, 50, 25, 150, 5]
    }
    return pd.DataFrame(data)

def test_lower(df_products):
    mask1 = parse_and_filter(df_products, "price < 50")
    assert sorted(df_products[mask1].product_id) == [102, 103, 105]

def test_exploit_fails(df_products):
    with pytest.raises(ValueError) as e:
        mask1 = parse_and_filter(df_products,
            """price < 50 and @pd.core.frame.com.builtins.__import__("os").system("/bin/echo password")""")
    assert 'Invalid filter syntax' in str(e)

@pytest.mark.parametrize('expression,ids', [
    ("price < 50 and category == 'Electronics'", [103]),
    ("stock < 100 or category == 'Home Goods'", [101, 103, 104, 106]),
    ("(price > 100 and stock < 20) or category == 'Books'", [101, 102, 105, 106]),
])
def test_logical_chaining(df_products, expression, ids):
    mask1 = parse_and_filter(df_products, expression)
    assert sorted(df_products[mask1].product_id) == sorted(ids)

def test_in_op(df_products):
    mask1 = parse_and_filter(df_products, "product_id in [101, 102]")
    assert sorted(df_products[mask1].product_id) == [101, 102]

