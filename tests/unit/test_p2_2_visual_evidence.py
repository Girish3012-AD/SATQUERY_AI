import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from shapely.geometry import Polygon

from src.executor.flood_specialist import FloodSpecialist
from src.executor.bi_temporal_water_specialist import BiTemporalWaterChangeSpecialist
from src.executor.building_specialist import BuildingDetectionSpecialist
from src.executor.change_specialist import ChangeSpecialist
from src.executor.sar_specialist import SARSpecialist
from src.schemas.evidence import Evidence

def create_dummy_raster(path: Path, data: np.ndarray, nodata=0, crs="EPSG:32645"):
    transform = from_origin(300000, 3100000, 10, 10)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return transform

def test_water_overlay_geometry(tmp_path):
    """Test FloodSpecialist produces valid geometry for water overlay."""
    spec = FloodSpecialist()
    t1 = tmp_path / "B03.tif"
    t2 = tmp_path / "B08.tif"
    
    # Create synthetic data where NDWI > 0 (water)
    # NDWI = (G - NIR) / (G + NIR)
    # For water: G > NIR. Let's make a 10x10 block of water in a 20x20 image.
    green = np.full((20, 20), 1500, dtype=np.uint16)
    nir = np.full((20, 20), 2000, dtype=np.uint16)
    
    # Water block
    green[5:15, 5:15] = 2000
    nir[5:15, 5:15] = 500
    
    create_dummy_raster(t1, green)
    create_dummy_raster(t2, nir)
    
    ev = spec.infer([str(t1), str(t2)], parameters={"ndwi_threshold": 0.0})
    assert ev.geometry is not None
    assert ev.geometry["type"] == "Polygon"
    assert "coordinates" in ev.geometry

def test_temporal_change_overlay_geometry(tmp_path):
    """Test ChangeSpecialist produces valid geometry for change overlay."""
    spec = ChangeSpecialist()
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    
    data1 = np.full((20, 20), 100, dtype=np.float32)
    data2 = np.full((20, 20), 100, dtype=np.float32)
    data2[5:10, 5:10] = 500 # Changed area
    
    create_dummy_raster(t1, data1)
    create_dummy_raster(t2, data2)
    
    ev = spec.infer([str(t1), str(t2)], parameters={"threshold": 0.5})
    assert ev.geometry is not None
    assert ev.geometry["type"] == "Polygon"
    assert "coordinates" in ev.geometry

def test_bi_temporal_water_change_overlay_geometry(tmp_path):
    """Test BiTemporalWaterChangeSpecialist produces geometry of T2 water state."""
    spec = BiTemporalWaterChangeSpecialist()
    t1 = tmp_path / "bitemp_t1.tif"
    t2 = tmp_path / "bitemp_t2.tif"
    
    # Since BiTemporalWaterChangeSpecialist reads green and nir from a single multi-band raster
    data1 = np.zeros((2, 20, 20), dtype=np.uint16)
    data2 = np.zeros((2, 20, 20), dtype=np.uint16)
    
    # T1: no water (G < NIR)
    data1[0, :, :] = 1000 # Green
    data1[1, :, :] = 2000 # NIR
    
    # T2: water in center (G > NIR)
    data2[0, :, :] = 1000
    data2[1, :, :] = 2000
    data2[0, 5:15, 5:15] = 2000
    data2[1, 5:15, 5:15] = 500
    
    # Write multi-band
    transform = from_origin(300000, 3100000, 10, 10)
    for p, d in [(t1, data1), (t2, data2)]:
        with rasterio.open(
            p, 'w', driver='GTiff', height=20, width=20, count=2,
            dtype=d.dtype, crs="EPSG:32645", transform=transform
        ) as dst:
            dst.write(d)
            
    ev = spec.infer([str(t1), str(t2)], parameters={"green_band": 1, "nir_band": 2, "ndwi_threshold": 0.0})
    assert ev.geometry is not None
    assert ev.geometry["type"] in ("Polygon", "MultiPolygon")
    assert "coordinates" in ev.geometry

def test_building_overlay_geometry(tmp_path, monkeypatch):
    """Test BuildingDetectionSpecialist produces bounding boxes/polygons."""
    spec = BuildingDetectionSpecialist(checkpoint_path="dummy")
    t1 = tmp_path / "build_t1.tif"
    create_dummy_raster(t1, np.full((20, 20), 100, dtype=np.uint16))
    
    def mock_predict(*args, **kwargs):
        class DummyCRS:
            def to_string(self): return "EPSG:32645"
            
        return {
            "mask": np.zeros((20, 20), dtype=np.uint8),
            "probability": np.zeros((20, 20), dtype=np.float32),
            "transform": from_origin(300000, 3100000, 10, 10),
            "crs": DummyCRS(),
            "covered_width": 20,
            "covered_height": 20,
        }
    monkeypatch.setattr("src.executor.building_specialist.predict_building_raster", mock_predict)
    monkeypatch.setattr("src.executor.building_specialist.BuildingDetectionSpecialist._get_model", lambda *args, **kwargs: None)
    
    ev = spec.infer([str(t1)], parameters={"confidence_threshold": 0.5})
    assert ev.geometry is not None
    assert ev.geometry["type"] in ("Polygon", "MultiPolygon")
    assert "coordinates" in ev.geometry

def test_sar_no_fabricated_geometry(tmp_path):
    """Test SARSpecialist does NOT fabricate geometry for statistical output."""
    spec = SARSpecialist()
    t1 = tmp_path / "sar.tif"
    create_dummy_raster(t1, np.full((20, 20), 100, dtype=np.float32))
    
    ev = spec.infer([str(t1)])
    assert ev.geometry is None  # Must not fabricate geometry

def test_missing_geometry_graceful():
    """Test that missing geometry is gracefully handled in Evidence."""
    ev = Evidence(
        evidence_id="EVID-TEST",
        source="Test",
        task="test_task",
        model="TestModel",
        modality="optical",
        geometry=None,
        confidence=0.9
    )
    assert ev.geometry is None
    d = ev.model_dump()
    assert d["geometry"] is None

