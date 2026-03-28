import pytest
from tools.stock_data import _analyze_stock_internal
from tools.encode_image_to_base64 import encode_image_to_base64
from shared_store import BASE64_STORE

def test_stock_data_returns_keys():
    result = _analyze_stock_internal("AAPL", "1mo")
    assert "current_price" in result
    assert "rsi" in result
    assert "trend" in result

def test_stock_invalid_ticker():
    result = _analyze_stock_internal("INVALIDXYZ123", "1mo")
    assert "error" in result
