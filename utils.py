"""DICOM loading, NIfTI export, and pseudonym generation."""

from __future__ import annotations
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


def _build_affine(datasets: list) -> np.ndarray:
    """Construct a NIfTI-RAS affine from DICOM orientation/position tags.

    The volume passed to nibabel must have shape (Rows, Cols, Slices).
    DICOM coordinates are in LPS; we convert to NIfTI-RAS by negating the
    first two rows of the affine (flip L→R and P→A).
    """
    ds = datasets[0]
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        ps  = [float(x) for x in ds.PixelSpacing]   # [row_spacing, col_spacing]

        row_cos = np.array(iop[:3])   # direction of increasing column index
        col_cos = np.array(iop[3:])   # direction of increasing row index
        normal  = np.cross(row_cos, col_cos)

        if len(datasets) > 1:
            ipp2    = np.array([float(x) for x in datasets[1].ImagePositionPatient])
            # Signed projection onto normal — preserves direction relative to slice order
            spacing = float(np.dot(ipp2 - ipp, normal))
        else:
            spacing = float(getattr(ds, "SliceThickness", 1.0) or 1.0)

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
