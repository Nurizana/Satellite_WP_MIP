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

# 1. Define Paths based on repository structure
csv_path = 'Data3/Hourly_PMSS_202401_WCoordinate.csv'
shapefile_zip = 'Data4/Malaysia_WGS1984.zip'
output_dir = 'Data2'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 2. Load the Data
print("Loading CSV and Shapefile...")
df = pd.read_csv(csv_path)

# Extract coordinates (assuming columns are named 'Longitude' and 'Latitude')
# Adjust column names if they differ in your CSV
points = df[['Longitude', 'Latitude']].values

# Load Malaysia boundary mask directly from the zip file
malaysia_gdf = gpd.read_file(f'zip://{shapefile_zip}')

# 3. Setup Interpolation Grid parameters for Malaysia
# Bounding box roughly: Lon 99°E to 120°E, Lat 0.5°N to 8°N
lon_min, lon_max = 99.0, 120.0
lat_min, lat_max = 0.5, 8.5
resolution = 0.05 # ~5km resolution, adjust for finer/coarser output

# Create meshgrid for interpolation
grid_lon, grid_lat = np.meshgrid(
    np.arange(lon_min, lon_max, resolution),
    np.arange(lat_max, lat_min, -resolution) # Top to bottom for raster
)

# 4. Identify Hourly Columns
# Assuming all columns except spatial/ID columns are hourly rainfall data
exclude_cols = ['Station', 'Longitude', 'Latitude', 'Station_ID']
hourly_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Found {len(hourly_cols)} hourly timesteps to process.")

# 5. Process each timestep
for i, col in enumerate(hourly_cols):
    print(f"Processing frame {i:03d} for column: {col}...")
    values = df[col].values
    
    # Interpolate using Linear (or 'cubic', 'nearest')
    grid_z = griddata(points, values, (grid_lon, grid_lat), method='linear')
    
    # Fill NaN values (outside interpolation hull) with 0 or a nodata value
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    
    # Define affine transform for GeoTIFF
    transform = from_origin(lon_min, lat_max, resolution, resolution)
    
    # Create a temporary unmasked GeoTIFF
    temp_tif = f'{output_dir}/temp_{i:03d}.tif'
    with rasterio.open(
        temp_tif, 'w', driver='GTiff',
        height=grid_z.shape[0], width=grid_z.shape[1],
        count=1, dtype=grid_z.dtype,
        crs='EPSG:4326', transform=transform, nodata=0.0
    ) as dst:
        dst.write(grid_z, 1)
        
    # 6. Mask the GeoTIFF with the Malaysia Shapefile
    with rasterio.open(temp_tif) as src:
        out_image, out_transform = mask(src, malaysia_gdf.geometry, crop=True, nodata=0.0)
        out_meta = src.meta.copy()
        
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Save the final masked GeoTIFF matching the HTML nomenclature
    final_tif = f'{output_dir}/map_timestep_{i:03d}.tif'
    with rasterio.open(final_tif, "w", **out_meta) as dest:
        dest.write(out_image)
        
    # Cleanup temporary file
    os.remove(temp_tif)

print("Interpolation and masking complete! All GeoTIFFs saved to Data2/.")