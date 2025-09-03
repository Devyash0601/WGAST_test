# %% [markdown]
# ## Tutorial 02: Preprocessing and Building Terra MODIS, Landsat8, and Sentinel-2 Triples
# 
# In this tutorial, we will prepare the data downloaded in Tutorial 01 to create clean and aligned image triples (Terra MODIS, Landsat, Sentinel-2) for spatio-temporal fusion tasks.
# 
# We will:
# - Load Terra MODIS, Landsat 8, and Sentinel-2 image collections matched on common dates.
# - Apply progressive focal mean interpolation to fill missing values due to clouds or technical issues.
# - Spatially align Terra MODIS images the Sentinel-2 resolution.
# - Save the resulting triples in a structured format.
# 
# 
# These processed triples can then be used to train and evaluate models such as WGAST or any custom deep learning architecture for spatio-temporal fusion of land surface temperature tasks.

# %%
import sys
import os
sys.path.append(os.path.abspath('..'))  # go up to root directory

from data_preparation.GetTriple import GetTriple
from data_preparation.DataProcessor import DataProcessor
import numpy as np
import time
import pandas as pd
from datetime import datetime

# %% [markdown]
# ## Step 01 : Load the Data for Triple Construction

# %%
# Initialize the triple preparation helper
get_triple = GetTriple()

# Load the common dates shared across all three satellite sources
common_dates_array = np.load('commun_dates.npy')

# Load Sentinel-2 images that match the common dates
sentinel2_images = get_triple.load_sentinel(
    'data/raw/Sentinel2',
    common_dates_array
)

# Load Landsat 8 images that match the common dates
landsat_images = get_triple.load_landsat(
    'data/raw/Landsat8',
    common_dates_array
)

# Load MODIS images that match the common dates
modis_images = get_triple.load_modis(
    'data/raw/MODIS',
    common_dates_array
)


# %% [markdown]
# ## Step 02 : Spatial Interpolation

# %%
# Initialize the data processor
data_processor = DataProcessor()

# Get target resolution dimensions from Sentinel-2 (highest resolution: 10 m)
height, width = sentinel2_images[0][0][0].shape

# (Optional) Get original Landsat resolution (30 m)
height2, width2 = landsat_images[0][0][0].shape

# --- Interpolation for Terra MODIS Images ---
# Apply progressive focal mean to fill missing values, then upscale to Sentinel-2 resolution
start_time = time.time()
modis_preprocessed = data_processor.progressive_focal_mean(modis_images)
modis_images_interpolated = data_processor.resize_modis_images(modis_preprocessed, height, width)
end_time = time.time()
print(f"Terra MODIS interpolation took {end_time - start_time:.2f} seconds.")


# --- Interpolation for Landsat Images ---
# Apply multiband progressive focal mean 
start_time = time.time()

landsat_proprocessed = data_processor.progressive_focal_mean_multiband(
    landsat_images, initial_size=5, step_size=5
)

end_time = time.time()
print(f"Landsat 8 interpolation took {end_time - start_time:.2f} seconds.")

# --- Interpolation for Sentinel-2 Images ---
# Sentinel-2 is already high resolution
start_time = time.time()

sentinel2_proprocessed = data_processor.progressive_focal_mean_multiband(
    sentinel2_images, initial_size=15, step_size=15
)

end_time = time.time()
print(f"Sentinel 2 interpolation took {end_time - start_time:.2f} seconds.")


# %% [markdown]
# ## Step 03 : Save the Triples

# %%
import os
from datetime import datetime

# Parse the dates from string to datetime objects
dates = [datetime.strptime(date, "%Y-%m-%d") for date in common_dates_array]

# Define output folder for the triple dataset
out_dir = "data/Triple/MODIS_Landsat8_Sentinel2"
os.makedirs(out_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Save the preprocessed Terra MODIS, Landsat 8, and Sentinel-2 images
get_triple.save_modis_formatted(modis_images_interpolated, dates, out_dir)
get_triple.save_landsat_formatted(landsat_proprocessed, dates, out_dir)
get_triple.save_sentinel_formatted(sentinel2_proprocessed, dates, out_dir)



