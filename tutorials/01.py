# %% [markdown]
# # Tutorial 01: Downloading Satellite Data from Google Earth Engine
# 
# In this tutorial, we will download and prepare Earth Observation data from three different satellite platforms using Google Earth Engine (GEE) and our custom Python processors:
# 
# - Sentinel-2 : NDVI, NDWI, NDBI (10–20 m, 5-day revisit).
# - Landsat 8 : LST, NDVI, NDWI, NDBI (30 m, 16-day).
# - Terra MODIS : LST (1 km, daily).
# 
# After retrieving the data, we will:
# - Filter images based on pixel quality and availability.
# - Extract only the relevant spectral indices or temperature bands.
# - Find the spatio-temporal intersection between the three satellite datasets.
# - Save the matched results for further processing and training of WGAST.
# 

# %%
import sys
import os
sys.path.append(os.path.abspath('..'))  # go up to root directory

from data_download.Sentinel2Processor import Sentinel2Processor
from data_download.Landsat8Processor import Landsat8Processor
from data_download.MODISProcessor import MODISProcessor


# %% [markdown]
# ## Step 01 : Define Region of Interest and Time Range
# The ROI and the start/end dates below are those used in the WGAST study. For other applications or locations, you should update these values accordingly.

# %%
roi = [85.23, 23.32, 85.35, 23.38]
start_date = '2013-01-01'
end_date = '2025-08-31'

# %% [markdown]
# ## Step 02: Download and Preprocess Satellite Data from GEE

# %%
# Define pixel availability threshold
# Only images with at least this percentage of valid (non-cloudy / non-missing) pixels will be kept
pixel_threshold = 80

# ------------------------------
# ✅ Sentinel-2
# ------------------------------

# Initialize processor
Sentinel_SR_preprocess = Sentinel2Processor(
    start_date=start_date,
    end_date=end_date,
    bounds=roi
)

# Load image collection
Sentinel2_data = Sentinel_SR_preprocess.get_Sentinel2_collection()
print('Sentinel-2 images before filtering:', Sentinel_SR_preprocess.countImages(Sentinel2_data))

# Filter by pixel availability
Sentinel2_data_filtered = Sentinel_SR_preprocess.filter_disponible_images(Sentinel2_data, pixel_threshold)
print('Sentinel-2 images after filtering:', Sentinel_SR_preprocess.countImages(Sentinel2_data_filtered))

# Extract NDVI, NDBI, NDWI
Sentinel2_data_filtered_SR = Sentinel_SR_preprocess.get_index(Sentinel2_data_filtered)

# Get timestamps
dates_sentinel2 = Sentinel_SR_preprocess.get_times(Sentinel2_data_filtered_SR)


# ------------------------------
# ✅ Landsat 8 
# ------------------------------

# Initialize processor
Landsat_LST_preprocess = Landsat8Processor(
    start_date=start_date,
    end_date=end_date,
    bounds=roi
)

# Load image collection
L8_data = Landsat_LST_preprocess.get_Landsat_collection()
print('Landsat 8 images before filtering:', Landsat_LST_preprocess.countImages(L8_data))

# Filter by pixel availability
L8_data_filtered = Landsat_LST_preprocess.filter_disponible_images(L8_data, pixel_threshold)
print('Landsat 8 images after filtering:', Landsat_LST_preprocess.countImages(L8_data_filtered))

# Extract LST, NDVI, NDBI, NDWI
L8_LST_index_data_filtered = Landsat_LST_preprocess.get_LST_index(L8_data_filtered)

# Get timestamps
L8_times_filtered = Landsat_LST_preprocess.get_times(L8_LST_index_data_filtered)


# ------------------------------
# ✅ MODIS 
# ------------------------------

# Initialize processor
MODIS_LST_preprocess = MODISProcessor(
    start_date=start_date,
    end_date=end_date,
    bounds=roi
)

# Load image collection
MODIS_data = MODIS_LST_preprocess.get_MODIS_collection()
print('MODIS images before filtering:', MODIS_LST_preprocess.countImages(MODIS_data))

# Filter by pixel availability
MODIS_data_filtered = MODIS_LST_preprocess.filter_disponible_images(MODIS_data, pixel_threshold)
print('MODIS images after filtering:', MODIS_LST_preprocess.countImages(MODIS_data_filtered))

# Extract LST band
MODIS_LST_data = MODIS_LST_preprocess.get_LST(MODIS_data_filtered)

# Get formatted timestamps
MODIS_times = MODIS_LST_preprocess.get_formatted_times(MODIS_data_filtered)


# %% [markdown]
# ##  Step 03: Find Common Acquisition Dates Across Sentinel-2, Landsat 8, and Terra MODIS

# %%
import numpy as np

# Extract date-only strings (YYYY-MM-DD) from each satellite time list
dates_modis = np.array([date.split(' ')[0] for date in MODIS_times])
dates_landsat = np.array([date.split('T')[0] for date in L8_times_filtered])
dates_sentinel = np.array([date.split('T')[0] for date in dates_sentinel2])

# Find the intersection across all three satellites
common_dates = np.intersect1d(dates_sentinel, dates_landsat)
common_dates = np.intersect1d(common_dates, dates_modis)

# Save the result for reproducibility
np.save('common_dates.npy', common_dates)
print(f"Total common dates found: {len(common_dates)}")

# Filter each dataset to keep only the common dates
MODIS_LST_data_common = MODIS_LST_preprocess.filter_by_common_dates(MODIS_LST_data, common_dates)
Landsat_LST_data_common = MODIS_LST_preprocess.filter_by_common_dates(L8_LST_index_data_filtered, common_dates)
Sentinel2_SR_data_common = MODIS_LST_preprocess.filter_by_common_dates(Sentinel2_data_filtered_SR, common_dates)


# %% [markdown]
# ## Step 04: Export Common-Date Satellite Images 

# %%
import os
import geemap

# ----------------- Export MODIS -----------------
modis_dir = 'data/raw/MODIS'
if not os.path.exists(modis_dir):
    os.makedirs(modis_dir)

geemap.ee_export_image_collection(
    MODIS_LST_data_common,
    out_dir=modis_dir,
    scale=1000,  # MODIS resolution: 1km
    region=MODIS_LST_preprocess.aoi
)


# ----------------- Export Landsat 8 -----------------
landsat_dir = 'data/raw/Landsat8'
if not os.path.exists(landsat_dir):
    os.makedirs(landsat_dir)

geemap.ee_export_image_collection(
    Landsat_LST_data_common,
    out_dir=landsat_dir,
    scale=30,  # Landsat 8resolution: 30m
    region=Landsat_LST_preprocess.aoi
)



# ----------------- Export Sentinel-2 -----------------
# 👉 If the size of Sentinel-2 images is small, prefer Option 1 (local export).
# 👉 If the size is too large, use Option 2 (Google Drive) to avoid memory or disk issues.

# ✅ Option 1: Export to local disk (recommended if image size is small)
sentinel_dir = 'data/raw/Sentinel2'
if not os.path.exists(sentinel_dir):
    os.makedirs(sentinel_dir)

geemap.ee_export_image_collection(
    Sentinel2_SR_data_common,
    out_dir=sentinel_dir,
    scale=10,  # Sentinel-2 resolution: 10m
    region=Sentinel_SR_preprocess.aoi
)

# ❌ Option 2: Export to Google Drive (uncomment below if local export is too heavy)
# geemap.ee_export_image_collection_to_drive(
#     Sentinel2_SR_data_common,
#     folder='export_sentinel_Indexs',
#     scale=10,
#     region=Sentinel_SR_preprocess.aoi,
#     maxPixels=1e13  # Increase limit for large export
# )


