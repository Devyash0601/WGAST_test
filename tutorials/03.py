# %% [markdown]
# # Tutorial 03: Structuring and Preparing the Dataset for Model Training
# 
# In this tutorial, we will organize the preprocessed satellite image triples (Terra MODIS, Landsat 8, Sentinel-2) into a clean and standardized format suitable for training spatio-temporal fusion models.
# 
# We will:
# - Generate valid (T1, T2) date pairs based on available image dates.
# - Create structured subfolders for each (T1, T2) pair.
# - Split the data into training and test sets.
# 
# This organized dataset can then be directly used to train deep learning models such as WGAST or any other spatio-temporal fusion network.

# %%
import numpy as np 
from datetime import datetime
import numpy as np 
import os
import shutil

# %% [markdown]
# ## Step 01 : Define Paired Dates (t1 and t2)

# %%
# Load two arrays of common valid dates
# You can change the arguments in np.load() to match the filenames you used when generating your (t1, t2) common dates.

common_dates_array1 = np.load(r'C:\Users\BIT\OneDrive - Birla Institute of Technology\Desktop\PROJECTS\Air-Quality\WGAST\data_download\common_dates_t1.npy', allow_pickle=True) # Dates for reference images (t1)
common_dates_array2 = np.load(r'C:\Users\BIT\OneDrive - Birla Institute of Technology\Desktop\PROJECTS\Air-Quality\WGAST\data_download\common_dates_t2.npy', allow_pickle=True)  # Dates for target images (t2)

# These two arrays must be generated beforehand by running your common date matching procedure twice:
# You are expected to ensure both arrays are aligned and consistent for further processing.

# Example: this was the array of (t1, t2) pairs used to train WGAST:
# np.array([
#     ['20170409', '20180223'],
#     ['20181021', '20190226'],
#     ['20190906', '20200401'],
#     ['20200722', '20200807'],
#     ['20220306', '20220322'],
#     ['20220813', '20220829'],
#     ['20230528', '20230613'],
#     ['20240412', '20240919'],
#     ['20240919', '20241005'],
#     ['20240919', '20241021']
# ], dtype='<U8')

# Convert strings to datetime objects
dates1 = [datetime.strptime(date, "%Y-%m-%d") for date in common_dates_array1]
dates2 = [datetime.strptime(date, "%Y-%m-%d") for date in common_dates_array2]


# Generate closest valid (t1, t2) pairs
# t1 = reference date, t2 = closest future date from the second array to represent the target date
result_dates = []

for d1 in dates1:
    # Exclude same-date match
    candidates = [d2 for d2 in dates2 if d2 != d1]
    
    # Prefer dates after d1 (future)
    after = [d2 for d2 in candidates if d2 > d1]
    if after:
        closest_date = min(after, key=lambda d2: (d2 - d1).days)
        result_dates.append((d1.strftime("%Y-%m-%d"), closest_date.strftime("%Y-%m-%d")))
    else:
        pass  # No future date available, ignore

# Format result: ['YYYYMMDD', 'YYYYMMDD']
dates = np.array([(start.replace('-', ''), end.replace('-', '')) for start, end in result_dates])

# %% [markdown]
# ## Step 02 : Prepare Dataset Structure

# %%
# Create the main directory for the division and subfolders for training and testing
main_folder = 'data/Tdivision'
os.makedirs(main_folder, exist_ok=True)

folders = ['train', 'test']
for folder in folders:
    os.makedirs(os.path.join(main_folder, folder), exist_ok=True)

# Split into training and test sets (adjust split_index as needed)
split_index = 7
train_dates = dates[:split_index]
test_dates = dates[split_index:]

# Copy (t1, t2) image pairs and their masks into structured folders
def create_subfolders(folder, date_pairs):
    for i in range(len(date_pairs)):
        pair_folder = os.path.join(main_folder, folder, f'pair_{i}')
        os.makedirs(pair_folder, exist_ok=True)

        date1, date2 = date_pairs[i]

        # t1
        modis_src = f'data/Triple/MODIS_Landsat8_Sentinel2/M_{date1}.tif'
        landsat_src = f'data/Triple/MODIS_Landsat8_Sentinel2/L_{date1}.tif'
        sentinel_src = f'data/Triple/MODIS_Landsat8_Sentinel2/S_{date1}.tif'

        modis_mask_src = f'data/Triple/MODIS_Landsat8_Sentinel2/M_mask_{date1}.npy'
        landsat_mask_src = f'data/Triple/MODIS_Landsat8_Sentinel2/L_mask_{date1}.npy'
        sentinel_mask_src = f'data/Triple/MODIS_Landsat8_Sentinel2/S_mask_{date1}.npy'

        j = 0
        modis_dst = os.path.join(pair_folder, f'{j:02d}_MODIS_{date1}.tif')
        landsat_dst = os.path.join(pair_folder, f'{j:02d}_Landsat_{date1}.tif')
        sentinel_dst = os.path.join(pair_folder, f'{j:02d}_Sentinel_{date1}.tif')

        modis_dst_mask = os.path.join(pair_folder, f'{j:02d}_MODIS_mask_{date1}.npy')
        landsat_dst_mask = os.path.join(pair_folder, f'{j:02d}_Landsat_mask_{date1}.npy')
        sentinel_dst_mask = os.path.join(pair_folder, f'{j:02d}_Sentinel_mask_{date1}.npy')

        shutil.copy(modis_src, modis_dst)
        shutil.copy(landsat_src, landsat_dst)
        shutil.copy(sentinel_src, sentinel_dst)

        shutil.copy(modis_mask_src, modis_dst_mask)
        shutil.copy(landsat_mask_src, landsat_dst_mask)
        shutil.copy(sentinel_mask_src, sentinel_dst_mask)

        # t2
        modis_src = f'data/Triple/MODIS_Landsat8_Sentinel2/M_{date2}.tif'
        landsat_src = f'data/Triple/MODIS_Landsat8_Sentinel2/L_{date2}.tif'

        modis_mask_src = f'data/Triple/MODIS_Landsat8_Sentinel2/M_mask_{date2}.npy'
        landsat_mask_src = f'data/Triple/MODIS_Landsat8_Sentinel2/L_mask_{date2}.npy'

        j = 1
        modis_dst = os.path.join(pair_folder, f'{j:02d}_MODIS_{date2}.tif')
        landsat_dst = os.path.join(pair_folder, f'{j:02d}_Landsat_{date2}.tif')

        modis_dst_mask = os.path.join(pair_folder, f'{j:02d}_MODIS_mask_{date2}.npy')
        landsat_dst_mask = os.path.join(pair_folder, f'{j:02d}_Landsat_mask_{date2}.npy')

        shutil.copy(modis_src, modis_dst)
        shutil.copy(landsat_src, landsat_dst)
        shutil.copy(modis_mask_src, modis_dst_mask)
        shutil.copy(landsat_mask_src, landsat_dst_mask)

create_subfolders('train', train_dates)
create_subfolders('test', test_dates)

print('Folders and files created successfully!')



