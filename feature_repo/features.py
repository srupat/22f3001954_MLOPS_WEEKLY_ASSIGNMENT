from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64


ROOT_DIR = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT_DIR / "feast_data" / "iris_features.parquet"

iris = Entity(
    name="iris",
    join_keys=["iris_id"],
    description="Unique identifier for each iris sample",
)

iris_source = FileSource(
    name="iris_source",
    path=str(PARQUET_PATH),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_features_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="iris_id", dtype=Int64),
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
    online=True,
    tags={"team": "mlops", "dataset": "iris"},
)
