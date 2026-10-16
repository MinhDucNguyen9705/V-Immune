"""BDW (burst descriptor word) schema, Parquet I/O and labeling-tool JSON export.

Implements report §2.1 (data model, spec §3.1). A BDW table is a pandas DataFrame whose columns are the
fields of :class:`BDW`; :data:`ARROW_SCHEMA` is the matching Arrow schema used for Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel


class SessionMeta(BaseModel):
    fs: float
    center_freq: float
    gain: float = 0.0
    antenna: str = "omni"
    distance_m: float | None = None
    environment: str = "sim"
    uav_type: str | None = None
    flight_mode: str | None = None
    session_id: str = "s0"
    node_id: str = "node-0"


class BDW(BaseModel):
    # signal fields
    toa_s: float
    tod_s: float
    pw_s: float
    fc_hz: float
    fc_off_norm: float
    bw_hz: float
    snr_db: float
    # provenance: pointer into the burst store + the snippet's sample rate / baseband center
    burst_iq_ref: str | None = None
    iq_fs: float | None = None
    iq_fc_hz: float | None = None
    # optional labels
    signal_type: str | None = None
    protocol: str | None = None
    source_id: str | None = None
    # session metadata (flattened, spec §3.1)
    fs: float
    center_freq: float
    gain: float = 0.0
    antenna: str = "omni"
    distance_m: float | None = None
    environment: str = "sim"
    uav_type: str | None = None
    flight_mode: str | None = None
    session_id: str = "s0"
    node_id: str = "node-0"


SIGNAL_FIELDS = ["toa_s", "tod_s", "pw_s", "fc_hz", "fc_off_norm", "bw_hz", "snr_db"]
SESSION_FIELDS = list(SessionMeta.model_fields)
COLUMNS = list(BDW.model_fields)

_ARROW_TYPES = {float: pa.float64(), str: pa.string()}


def _arrow_type(ann: Any) -> pa.DataType:
    for py, at in _ARROW_TYPES.items():
        if ann is py or py in getattr(ann, "__args__", ()):
            return at
    raise TypeError(ann)


ARROW_SCHEMA = pa.schema([pa.field(n, _arrow_type(f.annotation)) for n, f in BDW.model_fields.items()])


def empty_bdw() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64" if _arrow_type(BDW.model_fields[c].annotation) == pa.float64() else "object") for c in COLUMNS})


def make_bdw(signal: pd.DataFrame, session: SessionMeta, **labels: Any) -> pd.DataFrame:
    """Attach session metadata (and constant labels) to a frame holding the signal fields."""
    df = signal.copy()
    for k, v in session.model_dump().items():
        df[k] = v
    for k, v in labels.items():
        df[k] = v
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    df["fc_off_norm"] = (df["fc_hz"] - df["center_freq"]) / df["fs"]
    return df


def write_bdw(df: pd.DataFrame, path: str | Path) -> None:
    """Write a BDW frame to Parquet with :data:`ARROW_SCHEMA` (extra columns are kept)."""
    extra = [c for c in df.columns if c not in COLUMNS]
    schema = ARROW_SCHEMA
    for c in extra:
        schema = schema.append(pa.field(c, pa.Table.from_pandas(df[[c]], preserve_index=False).schema.field(c).type))
    table = pa.Table.from_pandas(df[COLUMNS + extra], schema=schema, preserve_index=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_bdw(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def validate(df: pd.DataFrame, n: int = 5) -> None:
    """Validate the first ``n`` rows against the pydantic model (cheap sanity check)."""
    for rec in df[COLUMNS].head(n).to_dict("records"):
        BDW.model_validate({k: (None if pd.isna(v) else v) if not isinstance(v, str) else v for k, v in rec.items()})


def to_labeling_json(df: pd.DataFrame) -> dict[str, Any]:
    """Export to the labeling-tool layout: ``{"session": {...}, "bdw": [flat dict per burst]}``."""
    sess = {k: _py(df[k].iloc[0]) for k in SESSION_FIELDS} if len(df) else {}
    rows = [{k: _py(v) for k, v in r.items()} for r in df[[c for c in COLUMNS if c not in SESSION_FIELDS]].to_dict("records")]
    return {"session": sess, "bdw": rows}


def write_labeling_json(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_labeling_json(df), indent=1))


def _py(v: Any) -> Any:
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v.item() if hasattr(v, "item") else v
