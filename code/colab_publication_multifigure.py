# Publication figure generator for Google Colab
!pip install geopandas matplotlib pandas pyogrio shapely fiona

from pathlib import Path
import sys

ROOT = Path('/content/New project/backend')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

!python /content/New\ project/backend/scripts/generate_publication_multifigure.py \
  --weighted-dir '/content/New project/backend/outputs/pop_building_weighted_rp' \
  --rp-dir '/content/New project/backend/outputs/rp_fire_response' \
  --output-figure '/content/New project/backend/outputs/pop_building_weighted_rp/figure_publication_multifigure.png'
