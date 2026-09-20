"""
Unit tests for the resilient _download_band downloader mechanism in rasuwa_flood_e2e.py.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from requests.exceptions import ChunkedEncodingError, RequestException
import rasterio

# We need to import the function, but since it's in a top-level script, 
# we can import it dynamically or by modifying sys.path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rasuwa_flood_e2e import _download_band

@pytest.fixture
def mock_rasterio():
    with patch("rasterio.open") as mock_open:
        yield mock_open

@pytest.fixture
def mock_sign_href():
    with patch("rasuwa_flood_e2e._sign_href", return_value="https://signed.url") as mock_sign:
        yield mock_sign

def test_fresh_download(tmp_path, mock_rasterio, mock_sign_href):
    dest = tmp_path / "test.tif"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "100"}
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    
    with patch("requests.get", return_value=mock_response):
        _download_band("dummy", dest, max_retries=1)
        
    assert dest.exists()
    assert dest.read_bytes() == b"chunk1chunk2"

def test_interrupted_download_resume(tmp_path, mock_rasterio, mock_sign_href):
    dest = tmp_path / "test.tif"
    
    # Attempt 1: throws ChunkedEncodingError after writing chunk1
    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.headers = {"content-length": "12"}
    
    def iter_fail(*args, **kwargs):
        yield b"chunk1"
        raise ChunkedEncodingError("Interrupted")
    
    mock_resp1.iter_content = iter_fail
    
    # Attempt 2: responds with 206 Partial Content, writes chunk2
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 206
    mock_resp2.headers = {"content-length": "6"}
    mock_resp2.iter_content.return_value = [b"chunk2"]
    
    with patch("requests.get", side_effect=[mock_resp1, mock_resp2]) as mock_get:
        with patch("time.sleep"):  # skip sleep
            _download_band("dummy", dest, max_retries=2)
            
    assert dest.exists()
    assert dest.read_bytes() == b"chunk1chunk2"
    assert mock_get.call_count == 2
    # Verify the second call included the Range header
    args, kwargs = mock_get.call_args_list[1]
    assert kwargs["headers"]["Range"] == "bytes=6-"

def test_already_complete_file(tmp_path, mock_rasterio, mock_sign_href):
    dest = tmp_path / "test.tif"
    dest.write_bytes(b"completed_file_content")
    
    mock_response = MagicMock()
    mock_response.status_code = 416  # Range not satisfiable
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        _download_band("dummy", dest, max_retries=1)
        
    assert dest.read_bytes() == b"completed_file_content"

def test_retry_after_connection_failure(tmp_path, mock_rasterio, mock_sign_href):
    dest = tmp_path / "test.tif"
    
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.headers = {"content-length": "6"}
    mock_resp2.iter_content.return_value = [b"chunk1"]
    
    with patch("requests.get", side_effect=[ConnectionError("Boom"), mock_resp2]):
        with patch("time.sleep"):
            _download_band("dummy", dest, max_retries=2)
            
    assert dest.read_bytes() == b"chunk1"

def test_failure_after_maximum_retries(tmp_path, mock_rasterio, mock_sign_href):
    dest = tmp_path / "test.tif"
    
    with patch("requests.get", side_effect=ConnectionError("Boom")):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Download failed after 2 attempts"):
                _download_band("dummy", dest, max_retries=2)
