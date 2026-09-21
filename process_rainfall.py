import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from scipy.interpolate import griddata
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Define Paths
csv_path = 'Data3/Hourly_PMSS_202401_WCoordinate.csv'
shapefile_zip = 'Data4/Malaysia_WGS1984.zip'
output_dir = 'Data2'

os.makedirs(output_dir, exist_ok=True)

# 2. Load the Data
print("Loading CSV and Shapefile...")
df = pd.read_csv(csv_path)

# Convert RAIN to numeric (forces any text like "Trace" to NaN), then drop invalid rows
df['RAIN'] = pd.to_numeric(df['RAIN'], errors='coerce')
df = df.dropna(subset=['RAIN', 'LATITUDE', 'LONGITUDE'])

# Load Malaysia boundary mask
malaysia_gdf = gpd.read_file(f'zip://{shapefile_zip}')

# 3. Setup Interpolation Grid parameters
lon_min, lon_max = 99.0, 120.0
lat_min, lat_max = 0.5, 8.5
resolution = 0.05 

grid_lon, grid_lat = np.meshgrid(
    np.arange(lon_min, lon_max, resolution),
    np.arange(lat_max, lat_min, -resolution) 
)

# 4. Group by Time (DAY and HOUR) and process each timestep
time_groups = df.groupby(['DAY', 'HOUR'])
frame_index = 0
total_required_frames = 124

for (day, hour), group in time_groups:
    if frame_index >= total_required_frames:
        break
        
    print(f"Processing frame {frame_index:03d} (Day {int(day)}, Hour {int(hour)})...")
    
    # Extract coordinates and rainfall for this specific hour
    points = group[['LONGITUDE', 'LATITUDE']].values
    values = group['RAIN'].values
    
    # Skip if there aren't enough points to interpolate
    if len(points) < 3:
        print("Skipping - insufficient data points.")
        frame_index += 1
        continue

    # Interpolate
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    
    transform = from_origin(lon_min, lat_max, resolution, resolution)
    
    # Create temporary GeoTIFF
    temp_tif = f'{output_dir}/temp_{frame_index:03d}.tif'
    with rasterio.open(
        temp_tif, 'w', driver='GTiff',
        height=grid_z.shape[0], width=grid_z.shape[1],
        count=1, dtype=grid_z.dtype,
        crs='EPSG:4326', transform=transform, nodata=0.0
    ) as dst:
        dst.write(grid_z, 1)
        
    # Mask with Malaysia Shapefile
    with rasterio.open(temp_tif) as src:
        out_image, out_transform = mask(src, malaysia_gdf.geometry, crop=True, nodata=0.0)
        out_meta = src.meta.copy()
        
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Save final masked file
    final_tif = f'{output_dir}/map_timestep_{frame_index:03d}.tif'
    with rasterio.open(final_tif, "w", **out_meta) as dest:
        dest.write(out_image)
        
    os.remove(temp_tif)
    frame_index += 1

print("Interpolation and masking complete! All GeoTIFFs saved to Data2/.")