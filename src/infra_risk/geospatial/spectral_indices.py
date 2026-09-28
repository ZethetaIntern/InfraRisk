"""
Spectral Index Calculator for Sentinel-2 Imagery
Computes NDVI, NDBI, NDWI, MNDWI, BSI, NBR from surface reflectance bands
"""
import numpy as np
from typing import Dict, Optional
from src.infra_risk.schemas.geospatial import SatelliteImage, SpectralIndices, SpectralBand
from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class SpectralIndexCalculator:
    """Calculates spectral indices from Sentinel-2 surface reflectance"""
    
    def __init__(self):
        # Band indices for Sentinel-2
        self.band_map = {
            SpectralBand.B2: 0,   # Blue
            SpectralBand.B3: 1,   # Green
            SpectralBand.B4: 2,   # Red
            SpectralBand.B5: 3,   # Red Edge 1
            SpectralBand.B6: 4,   # Red Edge 2
            SpectralBand.B7: 5,   # Red Edge 3
            SpectralBand.B8: 6,   # NIR
            SpectralBand.B8A: 7,  # Narrow NIR
            SpectralBand.B11: 8,  # SWIR 1
            SpectralBand.B12: 9,  # SWIR 2
        }
    
    def compute_all(self, image: SatelliteImage) -> SpectralIndices:
        """Compute all spectral indices from image bands"""
        bands = image.bands
        
        # Extract required bands
        nir = bands.get(SpectralBand.B8)
        red = bands.get(SpectralBand.B4)
        green = bands.get(SpectralBand.B3)
        blue = bands.get(SpectralBand.B2)
        swir1 = bands.get(SpectralBand.B11)
        swir2 = bands.get(SpectralBand.B12)
        red_edge1 = bands.get(SpectralBand.B5)
        red_edge2 = bands.get(SpectralBand.B6)
        red_edge3 = bands.get(SpectralBand.B7)
        nir_narrow = bands.get(SpectralBand.B8A)
        
        if nir is None or red is None:
            raise ValueError("Required bands (NIR, Red) not available")
        
        # Compute indices
        ndvi = self._safe_divide(nir - red, nir + red)
        ndbi = self._compute_ndbi(swir1, nir)
        ndwi = self._safe_divide(green - nir, green + nir) if green is not None else None
        mndwi = self._safe_divide(green - swir1, green + swir1) if green is not None and swir1 is not None else None
        bsi = self._compute_bsi(swir1, red, nir, blue) if swir1 is not None and blue is not None else None
        nbr = self._safe_divide(nir - swir2, nir + swir2) if swir2 is not None else None
        
        return SpectralIndices(
            ndvi=ndvi,
            ndbi=ndbi,
            ndwi=ndwi,
            mndwi=mndwi,
            bsi=bsi,
            nbr=nbr,
            ndvi_mean=float(np.nanmean(ndvi)) if ndvi is not None else 0,
            ndvi_std=float(np.nanstd(ndvi)) if ndvi is not None else 0,
            ndbi_mean=float(np.nanmean(ndbi)) if ndbi is not None else 0,
            ndbi_std=float(np.nanstd(ndbi)) if ndbi is not None else 0,
        )
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division avoiding division by zero"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(numerator, denominator)
            result[denominator == 0] = np.nan
        return result
    
    def _compute_ndbi(self, swir1: Optional[np.ndarray], nir: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Built-up Index"""
        if swir1 is None:
            return np.full_like(nir, np.nan)
        return self._safe_divide(swir1 - nir, swir1 + nir)
    
    def _compute_bsi(
        self,
        swir1: Optional[np.ndarray],
        red: np.ndarray,
        nir: np.ndarray,
        blue: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute Bare Soil Index"""
        if swir1 is None or blue is None:
            return np.full_like(red, np.nan)
        numerator = (swir1 + red) - (nir + blue)
        denominator = (swir1 + red) + (nir + blue)
        return self._safe_divide(numerator, denominator)
    
    def compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Compute NDVI from NIR and Red bands"""
        return self._safe_divide(nir - red, nir + red)
    
    def compute_ndbi(self, swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Compute NDBI from SWIR1 and NIR bands"""
        return self._safe_divide(swir1 - nir, swir1 + nir)
    
    def compute_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Compute NDWI from Green and NIR bands"""
        return self._safe_divide(green - nir, green + nir)
    
    def compute_mndwi(self, green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """Compute MNDWI from Green and SWIR1 bands"""
        return self._safe_divide(green - swir1, green + swir1)
    
    def compute_bsi(
        self,
        swir1: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        blue: np.ndarray,
    ) -> np.ndarray:
        """Compute BSI from SWIR1, Red, NIR, Blue bands"""
        numerator = (swir1 + red) - (nir + blue)
        denominator = (swir1 + red) + (nir + blue)
        return self._safe_divide(numerator, denominator)
    
    def compute_nbr(self, nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """Compute NBR from NIR and SWIR2 bands"""
        return self._safe_divide(nir - swir2, nir + swir2)
    
    def compute_evi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        blue: np.ndarray,
        g: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        L: float = 1.0,
    ) -> np.ndarray:
        """Compute Enhanced Vegetation Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = g * (nir - red) / (nir + c1 * red - c2 * blue + L)
            evi[denominator == 0] = np.nan
        return evi
    
    def compute_savi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        L: float = 0.5,
    ) -> np.ndarray:
        """Compute Soil Adjusted Vegetation Index"""
        with np.errstate(divide='ignore', invalid='ignore'):
            savi = (nir - red) / (nir + red + L) * (1 + L)
            savi[denominator == 0] = np.nan
        return savi


def calculate_indices_from_array(
    bands: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Convenience function to calculate indices from band arrays"""
    calc = SpectralIndexCalculator()
    
    # Create minimal SatelliteImage
    image = SatelliteImage(
        image_id="temp",
        acquisition_date=None,
        cloud_cover_pct=0,
        geometry={},
        bands={k: v for k, v in bands.items() if k in SpectralBand.__members__},
    )
    
    indices = calc.compute_all(image)
    
    return {
        "ndvi": indices.ndvi,
        "ndbi": indices.ndbi,
        "ndwi": indices.ndwi,
        "mndwi": indices.mndwi,
        "bsi": indices.bsi,
        "nbr": indices.nbr,
    }