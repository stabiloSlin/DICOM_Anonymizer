"""DICOM loading, NIfTI export, and pseudonym generation."""

from __future__ import annotations
import csv
import datetime
import hashlib
import random
from pathlib import Path

import numpy as np
import pydicom
import nibabel as nib

# ── Pseudonym generation ──────────────────────────────────────────────────────

_ADJECTIVES = [
    "Swift", "Calm", "Bold", "Clear", "Bright", "Silent", "Sharp", "Deep",
    "Free", "Keen", "Pure", "Quiet", "Wise", "Cool", "Fair", "Kind",
    "Safe", "Soft", "Wild", "Warm", "Brave", "Fresh", "Grand", "Lean",
    "Mild", "Neat", "Rich", "Slim", "Strong", "Tall", "Wide", "Dark",
    "Smooth", "Stern", "Noble", "Brisk", "Crisp", "Fleet", "Stout", "Tidy",
    
]

_NOUNS = [
    "River", "Stone", "Cedar", "Maple", "Falcon", "Harbor", "Meadow",
    "Summit", "Ridge", "Valley", "Brook", "Forest", "Garden", "Grove",
    "Hill", "Lake", "Ocean", "Peak", "Plain", "Shore", "Spring", "Stream",
    "Tide", "Trail", "Wave", "Wood", "Canyon", "Cliff", "Coast", "Crest",
    "Delta", "Glen", "Heath", "Inlet", "Isle", "Knoll", "Mesa", "Pass",
    "Pond", "Reef", "Sand", "Slope", "Dune", "Mound", "Birch", "Hazel",
    "Trout", "Salmon", "Bass", "Pike", "Carp", "Perch", "Walleye",
    "Cod", "Herring", "Mackerel", "Sardine", "Anchovy", "Grouper", "Snapper",
    "Tuna", "Marlin", "Swordfish", "Flounder", "Halibut", "Tilapia", "Catfish",
    "Sturgeon", "Barracuda", "Eel", "Garfish", "MahiMahi", "Sailfish", "Trout",
    "Bluegill", "Crappie", "Dorado", "Perch", "Wahoo", "Zander","Yellowtail",
    "Albacore", "Amberjack", "Butterfish", "Cobia", "Dorado", "Escolar", "Grouper",
    "Zander","Hecht","Schmerle",
]


def generate_name(seed: str | None = None) -> str:
    """Return a short pseudonym like 'SwiftRiver'.

    If seed is given the result is deterministic (same seed → same name).
    """
    if seed:
        h = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(h)
    else:
        rng = random.Random()
    return rng.choice(_ADJECTIVES) + rng.choice(_NOUNS)


# ── DICOM loading ─────────────────────────────────────────────────────────────

_SKIP_NAMES = {"DICOMDIR", "DICOMDIR.dcm", ".DS_Store"}

def _find_dicom_files(path: Path) -> list[Path]:
    """Return all DICOM image candidate files under *path* recursively.

    Strategy:
    1. Collect files with known DICOM extensions (.dcm / .ima).
    2. Also collect all extension-less files (common on CD/DVD exports where
       every file is a DICOM slice with no suffix).
    3. Skip DICOMDIR index files and OS metadata.
    The two sets are merged and de-duplicated.
    """
    def _keep(f: Path) -> bool:
        return f.is_file() and f.name not in _SKIP_NAMES

    by_ext: set[Path] = set()
    for pat in ("*.dcm", "*.DCM", "*.ima", "*.IMA"):
        by_ext.update(f for f in path.rglob(pat) if _keep(f))

    # Files with no suffix — typical on DICOM CD/DVD media
    no_ext: set[Path] = {
        f for f in path.rglob("*") if _keep(f) and f.suffix == ""
    }

    return sorted(by_ext | no_ext)


def scan_dicom_folder(path: str | Path) -> list[dict]:
    """Read DICOM headers only (no pixel data) and group by SeriesInstanceUID.

    Returns a list of series dicts sorted by n_slices descending::

        [{"uid": ..., "description": ..., "modality": ...,
          "rows": ..., "cols": ..., "n_slices": ...}, ...]

    Returns an empty list for a single-file path.
    """
    path = Path(path)
    if not path.is_dir():
        return []

    candidates: list[Path] = _find_dicom_files(path)

    series: dict[str, dict] = {}
    for f in candidates:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "") or f"no_uid_{f.parent.name}")
            if uid not in series:
                series[uid] = {
                    "uid":         uid,
                    "description": str(getattr(ds, "SeriesDescription", "") or ""),
                    "modality":    str(getattr(ds, "Modality",           "") or ""),
                    "rows":        int(getattr(ds, "Rows",               0)  or 0),
                    "cols":        int(getattr(ds, "Columns",            0)  or 0),
                    "n_slices":    0,
                }
            series[uid]["n_slices"] += 1
        except Exception:
            continue

    return sorted(series.values(), key=lambda s: s["n_slices"], reverse=True)


def _tag_float(ds, name: str) -> float | None:
    """Return a DICOM float tag, or None if missing/unparseable."""
    try:
        v = getattr(ds, name, None)
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _tag_vec(ds, name: str) -> list[float] | None:
    """Return a DICOM multi-value tag as a list of floats, or None."""
    try:
        v = getattr(ds, name, None)
        if v is None:
            return None
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def load_dicom_series(
    path: str | Path,
    series_uid: str | None = None,
) -> tuple[np.ndarray, list, dict]:
    """Load a DICOM series from *path* (file or directory).

    Parameters
    ----------
    path       : file or directory to search
    series_uid : if given, only slices whose SeriesInstanceUID matches are loaded;
                 pass None to load all files (only safe when the folder contains
                 exactly one series, or when *path* is a single file)

    Returns
    -------
    volume   : ndarray, shape (Z, Y, X), rescaled to int16 HU / signal units
    datasets : list of pydicom Datasets (one per slice, sorted)
    meta     : dict with display metadata and window/level defaults
    """
    path = Path(path)
    candidates: list[Path] = _find_dicom_files(path) if path.is_dir() else [path]

    datasets: list = []
    for f in candidates:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if series_uid is not None:
                ds_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
                if ds_uid != series_uid:
                    continue
            _ = ds.pixel_array  # ensure pixel data is present
            datasets.append(ds)
        except Exception as e:
            print(f'Exception while loading {f}: {e}')
            continue

    print(f'Successfully loaded {len(datasets)} DICOM files'
          + (f' for series {series_uid}' if series_uid else ''))
    if not datasets:
        raise ValueError(f"No valid DICOM files with pixel data found in: {path}")

    datasets.sort(key=_slice_sort_key)

    slices = [_apply_rescale(ds, ds.pixel_array) for ds in datasets]
    volume = np.stack(slices, axis=0)
    print('Volume: ',volume.min(), volume.max(), volume.dtype)

    meta = _extract_metadata(datasets[0])
    meta["volume_shape"] = volume.shape
    meta["n_slices"] = volume.shape[0]

    # Window / level defaults from DICOM tags, fall back to data statistics
    ds0 = datasets[0]
    try:
        wc = float(ds0.WindowCenter[0] if hasattr(ds0.WindowCenter, "__iter__") else ds0.WindowCenter)
        ww = float(ds0.WindowWidth[0]  if hasattr(ds0.WindowWidth,  "__iter__") else ds0.WindowWidth)
    except (AttributeError, TypeError, ValueError):
        wc = float(np.percentile(volume, 50))
        ww = float(np.percentile(volume, 99) - np.percentile(volume, 1))
        ww = max(ww, 1.0)

    meta["window_center"] = wc
    meta["window_width"]  = ww
    meta["data_min"] = int(volume.min())
    meta["data_max"] = int(volume.max())

    # Affine mapping display index order (Z=slice, Y=row, X=col) → RAS mm.
    meta["display_affine"] = build_display_affine(datasets)

    # Per-axis voxel spacing in (X, Y, Z) = (col, row, slice) order, in mm.
    # Used for the image-coordinate readout: coordinate = voxel_index * spacing,
    # measured from the volume corner (matches external navigation devices).
    try:
        _, _, _, ps, sp, _ = _geometry(datasets)
        meta["voxel_spacing"] = (float(ps[1]), float(ps[0]), abs(float(sp)))
    except Exception:
        meta["voxel_spacing"] = (1.0, 1.0, 1.0)

    # ── DICOM geometry tags (for display + CSV export) ─────────────────────────
    ds_last = datasets[-1]
    meta["slice_thickness"]        = _tag_float(ds0, "SliceThickness")          # (0018,0050)
    meta["spacing_between_slices"] = _tag_float(ds0, "SpacingBetweenSlices")    # (0018,0088)
    ipp_first = _tag_vec(ds0,      "ImagePositionPatient")                       # (0020,0032)
    ipp_last  = _tag_vec(ds_last,  "ImagePositionPatient")
    meta["ipp_first"] = ipp_first
    meta["ipp_last"]  = ipp_last

    # True inter-slice spacing from the position tags = |IPP_last - IPP_first| / (N-1)
    n_sl = volume.shape[0]
    if ipp_first is not None and ipp_last is not None and n_sl > 1:
        meta["computed_slice_spacing"] = float(
            np.linalg.norm(np.array(ipp_last) - np.array(ipp_first)) / (n_sl - 1)
        )
    else:
        meta["computed_slice_spacing"] = None

    meta["source_path"] = str(path)
    meta["source_name"] = Path(path).name

    return volume, datasets, meta


def _apply_rescale(ds, arr: np.ndarray) -> np.ndarray:
    slope     = float(getattr(ds, "RescaleSlope",     1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    # Always go via float32 — direct uint16→int16 wraps values > 32767
    arr_f32 = arr.astype(np.float32) * slope + intercept
    return np.clip(arr_f32, -32768, 32767).astype(np.int16)


def _slice_sort_key(ds) -> float:
    # Project IPP onto the slice normal — correct for axial, coronal, sagittal
    try:
        iop    = [float(x) for x in ds.ImageOrientationPatient]
        normal = np.cross(np.array(iop[:3]), np.array(iop[3:]))
        ipp    = [float(x) for x in ds.ImagePositionPatient]
        return float(np.dot(normal, ipp))
    except Exception:
        pass
    try:
        return float(ds.InstanceNumber)
    except Exception:
        return 0.0


def _extract_metadata(ds) -> dict:
    def sg(attr: str, default: str = "") -> str:
        v = getattr(ds, attr, None)
        return str(v).strip() if v is not None else default

    return {
        "patient_name":       sg("PatientName",       "Unknown"),
        "patient_id":         sg("PatientID",          "Unknown"),
        "patient_dob":        sg("PatientBirthDate",   ""),
        "patient_sex":        sg("PatientSex",         ""),
        "patient_age":        sg("PatientAge",         ""),
        "study_date":         sg("StudyDate",          ""),
        "study_description":  sg("StudyDescription",  ""),
        "series_description": sg("SeriesDescription", ""),
        "modality":           sg("Modality",           ""),
        "manufacturer":       sg("Manufacturer",       ""),
    }


# ── NIfTI export ──────────────────────────────────────────────────────────────

_ANON_TAGS = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0010, 0x4000),  # PatientComments
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0032, 0x4000),  # StudyComments
]


def _geometry(datasets: list):
    """Return (row_cos, col_cos, normal, ps, spacing, ipp) from DICOM tags.

    row_cos : direction of increasing column index (iop[:3])
    col_cos : direction of increasing row index    (iop[3:])
    normal  : slice normal = row_cos × col_cos
    ps      : [row_spacing, col_spacing] in mm
    spacing : signed inter-slice spacing along the normal
    ipp     : ImagePositionPatient of the first slice (LPS mm)
    """
    ds  = datasets[0]
    iop = [float(x) for x in ds.ImageOrientationPatient]
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    ps  = [float(x) for x in ds.PixelSpacing]   # [row_spacing, col_spacing]

    row_cos = np.array(iop[:3])
    col_cos = np.array(iop[3:])
    normal  = np.cross(row_cos, col_cos)

    if len(datasets) > 1:
        ipp2    = np.array([float(x) for x in datasets[1].ImagePositionPatient])
        spacing = float(np.dot(ipp2 - ipp, normal))
    else:
        spacing = float(getattr(ds, "SliceThickness", 1.0) or 1.0)

    return row_cos, col_cos, normal, ps, spacing, ipp


def _build_affine(datasets: list) -> np.ndarray:
    """Construct a NIfTI-RAS affine from DICOM orientation/position tags.

    The volume passed to nibabel must have shape (Rows, Cols, Slices).
    DICOM coordinates are in LPS; we convert to NIfTI-RAS by negating the
    first two rows of the affine (flip L→R and P→A).
    """
    try:
        row_cos, col_cos, normal, ps, spacing, ipp = _geometry(datasets)

        # LPS affine: maps (row_idx, col_idx, slice_idx) → LPS mm
        affine_lps = np.eye(4)
        affine_lps[:3, 0] = col_cos * ps[0]   # dim-0 = rows  → col_cos direction
        affine_lps[:3, 1] = row_cos * ps[1]   # dim-1 = cols  → row_cos direction
        affine_lps[:3, 2] = normal  * spacing  # dim-2 = slices → normal (signed)
        affine_lps[:3, 3] = ipp

        # LPS → RAS: negate X and Y components (rows 0 and 1)
        lps_to_ras = np.diag([-1., -1., 1., 1.])
        return lps_to_ras @ affine_lps

    except Exception:
        return np.eye(4)


def build_display_affine(datasets: list) -> np.ndarray:
    """RAS affine for the *display* volume index order (Z=slice, Y=row, X=col).

    The viewer stacks slices on axis 0, so its volume has shape
    (n_slices, Rows, Cols) == (Z, Y, X). This affine maps a homogeneous
    index vector [Z, Y, X, 1] to Slicer-style RAS millimetres, following the
    LPS→RAS convention used throughout (3D Slicer stores coordinates in RAS).

    Returns the 4x4 identity if orientation tags are missing, so callers can
    always assume a usable matrix.
    """
    try:
        row_cos, col_cos, normal, ps, spacing, ipp = _geometry(datasets)

        # LPS affine for index order (slice, row, col):
        affine_lps = np.eye(4)
        affine_lps[:3, 0] = normal  * spacing  # dim-0 = slices (Z)
        affine_lps[:3, 1] = col_cos * ps[0]    # dim-1 = rows   (Y)
        affine_lps[:3, 2] = row_cos * ps[1]    # dim-2 = cols   (X)
        affine_lps[:3, 3] = ipp

        lps_to_ras = np.diag([-1., -1., 1., 1.])
        return lps_to_ras @ affine_lps

    except Exception:
        return np.eye(4)


def export_nifti(datasets: list, output_path: str | Path, patient_name: str) -> Path:
    """Write an anonymized NIfTI-1 file from *datasets*.

    All identifying DICOM metadata is stripped; only the pseudonym is stored
    in the NIfTI description field.
    """
    # Shape (Rows, Cols, Slices) — matches _build_affine's (row, col, slice) mapping
    volume = np.stack(
        [_apply_rescale(ds, ds.pixel_array) for ds in datasets], axis=2
    )

    affine = _build_affine(datasets)
    img    = nib.Nifti1Image(volume, affine)
    hdr    = img.header
    hdr.set_xyzt_units("mm", "sec")
    hdr.set_sform(affine, code=1)   # scanner RAS coordinates
    hdr.set_qform(affine, code=1)
    hdr["descrip"] = f"Anon:{patient_name}".encode()[:80]

    output_path = Path(output_path)
    nib.save(img, str(output_path))
    return output_path


# ── Geometry CSV export ────────────────────────────────────────────────────────

def _csv_scalar(v) -> str:
    return "" if v is None else str(v)


def _csv_vector(v) -> str:
    if not v:
        return ""
    return ";".join(f"{float(x):.6g}" for x in v)


def export_geometry_csv(meta: dict, out_dir: str | Path | None = None) -> Path:
    """Write the DICOM geometry tags of the loaded series to a new CSV file.

    A fresh, uniquely-named file is created on every call (timestamped), and it
    records the original file/folder name and the creation date alongside the
    tags. By default the CSV is written next to the loaded data; if that folder
    is not writable, it falls back to the current working directory.
    """
    src = Path(meta.get("source_path") or ".")
    if out_dir is None:
        out_dir = src if src.is_dir() else src.parent
    out_dir = Path(out_dir)

    ts = datetime.datetime.now()
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    base = meta.get("source_name") or "dicom"
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(base))
    fname = f"dicom_geometry_{safe}_{stamp}.csv"

    rows = [
        ("original_filename",                      _csv_scalar(meta.get("source_name"))),
        ("source_path",                            _csv_scalar(meta.get("source_path"))),
        ("created",                                ts.isoformat(timespec="seconds")),
        ("n_slices",                               _csv_scalar(meta.get("n_slices"))),
        ("SliceThickness_0018_0050",               _csv_scalar(meta.get("slice_thickness"))),
        ("SpacingBetweenSlices_0018_0088",         _csv_scalar(meta.get("spacing_between_slices"))),
        ("ImagePositionPatient_first_0020_0032",   _csv_vector(meta.get("ipp_first"))),
        ("ImagePositionPatient_last_0020_0032",    _csv_vector(meta.get("ipp_last"))),
        ("computed_slice_spacing_mm",              _csv_scalar(meta.get("computed_slice_spacing"))),
        ("pixel_spacing_xyz_mm",                   _csv_vector(meta.get("voxel_spacing"))),
    ]

    def _write(target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        p = target_dir / fname
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["field", "value"])
            w.writerows(rows)
        return p

    try:
        return _write(out_dir)
    except OSError:
        return _write(Path.cwd())
