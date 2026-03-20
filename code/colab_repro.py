# Colab-ready execution script
!pip install osmnx geopandas shapely pyproj folium networkx matplotlib pandas numpy

from pathlib import Path
import sys

ROOT = Path('/content/New project/backend')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

!python /content/New\ project/backend/scripts/run_population_building_weighted_rp_analysis.py \
  --data-root '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data' \
  --population-csv '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data/250_LOCAL_RESD_20260313.csv' \
  --building-shp '/content/drive/Othercomputers/내 Mac/Keywest_JetDrive 1/W2 Data/AL_D010_11_20260309/AL_D010_11_20260309.shp' \
  --rp-output-dir '/content/New project/backend/outputs/rp_fire_response' \
  --output-dir '/content/New project/backend/outputs/pop_building_weighted_rp'
