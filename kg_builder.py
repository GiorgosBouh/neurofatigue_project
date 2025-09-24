#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KG Builder for Neurofatigue multimodal dataset:
- Participants: data/01..12
- Sessions: per participant 01..03 (low/medium/high)
- Modalities: EEG, ECG, PPG, Resp, IMU
- Windowed features + Proxy labels (TIA_like)
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from tqdm import tqdm
import neurokit2 as nk

from neo4j import GraphDatabase
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------
# Config & Helpers
# -------------------------
# Load .env explicitly from the script's folder
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=str(ENV_PATH))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data")).expanduser()
WIN_SEC = float(os.getenv("WINDOW_SECONDS", "10"))
OVERLAP = float(os.getenv("WINDOW_OVERLAP", "0.5"))

# New tuning knobs (safe defaults for t2.micro)
MAX_PARTICIPANTS = int(os.getenv("MAX_PARTICIPANTS", "3"))
SESS_FILTER = os.getenv("SESS_FILTER", "").strip()  # "", "01", "02", or "03"
DOWNSAMPLE_HZ = float(os.getenv("DOWNSAMPLE_HZ", "0"))  # 0 disables
ENABLE_HRV = os.getenv("ENABLE_HRV", "1") == "1"        # 0 to disable HRV calc

# Fallback sampling rate if unknown
DEFAULT_FS = 256.0

# EEG bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

# Map session folder to label if needed
SESSION_MAP = {"01": "low", "02": "medium", "03": "high"}

# -------------------------
# Neo4j driver
# -------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def init_constraints():
    cypher = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Participant) REQUIRE p.pid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.sid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:SignalFile) REQUIRE f.fid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (w:TimeWindow) REQUIRE w.wid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Survey) REQUIRE q.qid IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Meta) REQUIRE m.mid IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (feat:Feature) ON (feat.code)",
        "CREATE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.type)"
    ]
    with driver.session(database=NEO4J_DATABASE) as s:
        for c in cypher:
            s.run(c)

# -------------------------
# IO & detection
# -------------------------
def detect_modality(fname: str) -> Optional[str]:
    f = fname.lower()
    if "eeg" in f:
        return "EEG"
    if "ecg" in f:
        return "ECG"
    if ("ppg" in f) or ("bvp" in f):
        return "PPG"
    if ("resp" in f) or ("respiration" in f) or ("breath" in f):
        return "RESP"
    if ("imu" in f) or ("acc" in f) or ("gyro" in f) or ("motion" in f) or ("wrist" in f):
        return "IMU"
    return None

def load_timeseries_csv(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    # get time vector
    t = None
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    elif "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    if t is None or not np.isfinite(t).any():
        fs = DEFAULT_FS
        t = np.arange(len(df)) / fs
    else:
        t = t[np.isfinite(t)]
        if t.size != len(df):
            fs = DEFAULT_FS
            t = np.arange(len(df)) / fs

    # estimate fs robustly
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = (1.0 / np.median(dt)) if dt.size else DEFAULT_FS

    # numeric channels (exclude time cols)
    chan_cols = [c for c in cols if c.lower() not in ["timestamp", "time"]]
    chans = {}
    for c in chan_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).sum() > 0:
            chans[c] = np.nan_to_num(vals)
    return np.asarray(t, dtype=float), chans, float(fs)

# -------------------------
# Downsampling helpers
# -------------------------
def maybe_downsample(x: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """Simple decimation if requested and fs > target."""
    if DOWNSAMPLE_HZ and fs > DOWNSAMPLE_HZ and len(x) > 0:
        factor = max(1, int(round(fs / DOWNSAMPLE_HZ)))
        if factor > 1:
            return x[::factor], fs / factor
    return x, fs

def maybe_downsample_stack(stack: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """Decimate along time axis for multi-channel arrays."""
    if DOWNSAMPLE_HZ and fs > DOWNSAMPLE_HZ and stack.shape[0] > 0:
        factor = max(1, int(round(fs / DOWNSAMPLE_HZ)))
        if factor > 1:
            return stack[::factor, :], fs / factor
    return stack, fs

# -------------------------
# Feature extraction
# -------------------------
def window_indices(t: np.ndarray, win_sec: float, overlap: float):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return []
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs_est = 1.0 / np.median(dt) if dt.size else DEFAULT_FS
    wlen = max(2, int(round(win_sec * fs_est)))
    hop = max(1, int(round(wlen * (1 - overlap))))
    idx = []
    N = t.size
    start = 0
    while start + wlen <= N:
        idx.append((start, start + wlen))
        start += hop
    return idx

def eeg_features(seg: np.ndarray, fs: float) -> Dict[str, float]:
    f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 1024))
    total_power = trapezoid(pxx, f) + 1e-12
    feats = {}
    for band, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = trapezoid(pxx[m], f[m]) if np.any(m) else 0.0
        feats[f"EEG_bp_{band}"] = float(bp)
        feats[f"EEG_rel_{band}"] = float(bp / total_power)
    feats["EEG_ratio_theta_alpha"] = feats["EEG_rel_theta"] / (feats["EEG_rel_alpha"] + 1e-12)
    p_norm = pxx / (pxx.sum() + 1e-12)
    feats["EEG_spectral_entropy"] = float(-(p_norm * np.log(p_norm + 1e-12)).sum())
    return feats

def imu_features(seg_xyz: np.ndarray, fs: float) -> Dict[str, float]:
    mag = np.linalg.norm(seg_xyz, axis=1)
    rms = float(np.sqrt(np.mean(mag**2)))
    jerk = np.diff(mag) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.0
    return {
        "IMU_rms": rms,
        "IMU_jerk_rms": jerk_rms,
        "IMU_var": float(np.var(mag))
    }

def hrv_from_ecg_ppg(sig: np.ndarray, fs: float) -> Dict[str, float]:
    try:
        _, info = nk.signal_rate(sig, sampling_rate=fs)
        peaks = info.get("peaks", None)
        if peaks is None or len(peaks) < 3:
            return {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_LFHF": np.nan}
        rr = np.diff(peaks) / fs
        hrv_time = nk.hrv_time(rr, show=False)
        rmssd = float(hrv_time.get("HRV_RMSSD", np.nan))
        sdnn = float(hrv_time.get("HRV_SDNN", np.nan))
        try:
            hrv_freq = nk.hrv_frequency(rr, show=False)
            lfhf = float(hrv_freq.get("HRV_LFHF", np.nan))
        except Exception:
            lfhf = np.nan
        return {"HRV_RMSSD": rmssd, "HRV_SDNN": sdnn, "HRV_LFHF": lfhf}
    except Exception:
        return {"HRV_RMSSD": np.nan, "HRV_SDNN": np.nan, "HRV_LFHF": np.nan}

def resp_features(seg: np.ndarray, fs: float) -> Dict[str, float]:
    try:
        rate = float(nk.resp_rate(seg, sampling_rate=fs)["Respiratory_Rate"].mean())
        instab = float(np.std(seg) / (np.mean(np.abs(seg)) + 1e-12))
        return {"RESP_rate": rate, "RESP_instability": instab}
    except Exception:
        return {"RESP_rate": np.nan, "RESP_instability": np.nan}

# -------------------------
# Proxy labels
# -------------------------
def z_score_series(values: List[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

def proxy_rules(dfw: pd.DataFrame) -> pd.DataFrame:
    df = dfw.copy()
    df["theta_alpha"] = df.get("EEG_ratio_theta_alpha", np.nan)
    df["specent"] = df.get("EEG_spectral_entropy", np.nan)
    df["alpha_rel"] = df.get("EEG_rel_alpha", np.nan)
    df["imu_rms"] = df.get("IMU_rms", np.nan)
    df["imu_jerk"] = df.get("IMU_jerk_rms", np.nan)
    df["rmssd"] = df.get("HRV_RMSSD", np.nan)

    def zcol(name):
        z = z_score_series(df[name].tolist())
        df[f"z_{name}"] = z
    for c in ["theta_alpha", "specent", "alpha_rel", "imu_rms", "imu_jerk", "rmssd"]:
        zcol(c)

    df["EEG_anomaly"] = ((df["z_theta_alpha"] > 2.0) |
                         (df["z_specent"] > 2.0) |
                         (df["z_alpha_rel"] < -2.0)).astype(int)

    df["Gait_instability"] = ((df["z_imu_rms"] > 2.0) |
                              (df["z_imu_jerk"] > 2.0)).astype(int)

    df["Stress_flag"] = (df["z_rmssd"] < -1.5).astype(int)

    stress_shift = df["Stress_flag"].rolling(window=3, center=True, min_periods=1).max().astype(int)

    df["TIA_like"] = ((df["EEG_anomaly"] == 1) &
                      (df["Gait_instability"] == 1) &
                      (stress_shift == 1)).astype(int)
    return df

# -------------------------
# Neo4j upserts
# -------------------------
def neo4j_merge_participant(tx, pid):
    tx.run("MERGE (p:Participant {pid:$pid})", pid=pid)

def neo4j_merge_session(tx, pid, sid, level):
    tx.run("""
    MERGE (s:Session {sid:$sid})
    SET s.level=$level
    WITH s
    MATCH (p:Participant {pid:$pid})
    MERGE (p)-[:HAS_SESSION]->(s)
    """, pid=pid, sid=sid, level=level)

def neo4j_merge_signalfile(tx, sid, fid, path, modality, fs):
    tx.run("""
    MERGE (f:SignalFile {fid:$fid})
    SET f.path=$path, f.modality=$modality, f.fs=$fs
    WITH f
    MATCH (s:Session {sid:$sid})
    MERGE (s)-[:HAS_SIGNAL]->(f)
    """, sid=sid, fid=fid, path=str(path), modality=modality, fs=fs)

def neo4j_merge_window_features(tx, wid, sid, t0, t1, feats: Dict[str, float]):
    tx.run("""
    MERGE (w:TimeWindow {wid:$wid})
    SET w.t0=$t0, w.t1=$t1
    WITH w
    MATCH (s:Session {sid:$sid})
    MERGE (s)-[:HAS_WINDOW]->(w)
    """, wid=wid, sid=sid, t0=float(t0), t1=float(t1))

    feat_rows = [{"wid": wid, "code": k, "value": float(v)} for k, v in feats.items() if np.isfinite(v)]
    if feat_rows:
        tx.run("""
        UNWIND $rows AS r
        MERGE (w:TimeWindow {wid:r.wid})
        MERGE (f:Feature {code:r.code})
        MERGE (w)-[:HAS_FEATURE]->(fv:FeatureValue {code:r.code, wid:r.wid})
        SET fv.value = r.value
        """, rows=feat_rows)

def neo4j_merge_events(tx, wid, flags: Dict[str, int]):
    rows = [{"wid": wid, "type": k, "flag": int(v)} for k, v in flags.items()]
    tx.run("""
    UNWIND $rows AS r
    MERGE (w:TimeWindow {wid:r.wid})
    MERGE (e:Event {type:r.type})
    MERGE (w)-[:HAS_EVENT]->(ev:EventFlag {type:r.type, wid:r.wid})
    SET ev.flag = r.flag
    """, rows=rows)

def neo4j_merge_survey(tx, pid, qid, payload: Dict):
    tx.run("""
    MERGE (q:Survey {qid:$qid})
    SET q += $data
    WITH q
    MATCH (p:Participant {pid:$pid})
    MERGE (p)-[:HAS_SURVEY]->(q)
    """, pid=pid, qid=qid, data=payload)

def neo4j_merge_metadata(tx, pid, mid, payload: Dict):
    tx.run("""
    MERGE (m:Meta {mid:$mid})
    SET m += $data
    WITH m
    MATCH (p:Participant {pid:$pid})
    MERGE (p)-[:HAS_META]->(m)
    """, pid=pid, mid=mid, data=payload)

# -------------------------
# Main build
# -------------------------
def normalize_pid(name: str) -> str:
    num = re.sub(r"\D", "", name)
    return f"P{int(num):02d}" if num else name

def normalize_sid(pid: str, sess_folder: str) -> str:
    label = SESSION_MAP.get(sess_folder, sess_folder)
    return f"{pid}:{label}"

def collect_surveys(data_root: Path) -> Dict[str, Dict]:
    out = {}
    pts = data_root / "pre_task_survey.xlsx"
    if pts.exists():
        df = pd.read_excel(pts)
        if "ID" in df.columns:
            for _, r in df.iterrows():
                try:
                    pid = f"P{int(r['ID']):02d}"
                except Exception:
                    continue
                out.setdefault(pid, {})
                out[pid]["pre_task"] = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
    pq = data_root / "preliminary_questionnaire.xlsx"
    if pq.exists():
        df = pd.read_excel(pq)
        id_col = None
        for c in df.columns:
            if str(c).strip().lower() in ["id", "participant", "participant_id"]:
                id_col = c
                break
        if id_col:
            for _, r in df.iterrows():
                try:
                    pid = f"P{int(r[id_col]):02d}"
                except Exception:
                    continue
                out.setdefault(pid, {})
                out[pid]["prelim"] = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
    return out

def collect_metadata(data_root: Path) -> Dict[str, Dict]:
    meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    out = {}
    pid_col = "participant_id" if "participant_id" in df.columns else df.columns[0]
    for _, r in df.iterrows():
        try:
            pid = f"P{int(r[pid_col]):02d}"
        except Exception:
            continue
        out[pid] = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
    return out

def build():
    import gc

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    init_constraints()
    surveys = collect_surveys(DATA_ROOT)
    meta = collect_metadata(DATA_ROOT)

    # Gather participants
    participants = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and re.match(r"^\d+$", p.name)])
    if MAX_PARTICIPANTS > 0:
        participants = participants[:MAX_PARTICIPANTS]

    for p_dir in tqdm(participants, desc="Participants"):
        pid = normalize_pid(p_dir.name)
        with driver.session(database=NEO4J_DATABASE) as s:
            s.execute_write(neo4j_merge_participant, pid)

        # attach survey/metadata if available
        with driver.session(database=NEO4J_DATABASE) as s:
            if pid in surveys:
                for k, payload in surveys[pid].items():
                    s.execute_write(neo4j_merge_survey, pid, f"{pid}:{k}", payload)
            if pid in meta:
                s.execute_write(neo4j_merge_metadata, pid, f"{pid}:meta", meta[pid])

        # sessions
        sess_dirs = sorted([q for q in p_dir.iterdir() if q.is_dir() and re.match(r"^\d+$", q.name)])
        if SESS_FILTER:
            sess_dirs = [q for q in sess_dirs if q.name == SESS_FILTER]

        for sess in sess_dirs:
            sid = normalize_sid(pid, sess.name)
            level = SESSION_MAP.get(sess.name, sess.name)
            with driver.session(database=NEO4J_DATABASE) as s:
                s.execute_write(neo4j_merge_session, pid, sid, level)

            files = sorted([f for f in sess.rglob("*.csv")])
            win_rows: List[Dict[str, float]] = []

            for fpath in files:
                modality = detect_modality(fpath.name)
                if modality is None:
                    continue

                # parse time series
                try:
                    t, chans, fs = load_timeseries_csv(fpath)
                except Exception as e:
                    print(f"[WARN] Failed to read {fpath}: {e}")
                    continue

                fid = f"{sid}:{fpath.name}"
                with driver.session(database=NEO4J_DATABASE) as s:
                    s.execute_write(neo4j_merge_signalfile, sid, fid, fpath, modality, fs)

                idx = window_indices(t, WIN_SEC, OVERLAP)
                if not idx or not np.isfinite(t).any():
                    continue
                N = len(t)

                if modality == "EEG":
                    if not chans:
                        continue
                    stack = np.vstack([v for v in chans.values()]).T  # [N, C]
                    stack, fs_eff = maybe_downsample_stack(stack, fs)
                    for (a, b) in idx:
                        if not (0 <= a < b <= stack.shape[0]):
                            continue
                        seg = np.nanmean(stack[a:b, :], axis=1)
                        feats = eeg_features(seg, fs_eff)
                        win_rows.append({"t0": float(t[a]), "t1": float(t[b-1]), **feats})

                elif modality == "IMU":
                    if not chans:
                        continue
                    stack = np.vstack([v for v in chans.values()]).T
                    stack, fs_eff = maybe_downsample_stack(stack, fs)
                    for (a, b) in idx:
                        if not (0 <= a < b <= stack.shape[0]):
                            continue
                        feats = imu_features(stack[a:b, :], fs_eff)
                        win_rows.append({"t0": float(t[a]), "t1": float(t[b-1]), **feats})

                elif modality in ["ECG", "PPG"]:
                    if not ENABLE_HRV or not chans:
                        pass
                    else:
                        ch = list(chans.values())[0]
                        ch, fs_eff = maybe_downsample(ch, fs)
                        for (a, b) in idx:
                            if not (0 <= a < b <= len(ch)):
                                continue
                            feats = hrv_from_ecg_ppg(ch[a:b], fs_eff)
                            win_rows.append({"t0": float(t[a]), "t1": float(t[b-1]), **feats})

                elif modality == "RESP":
                    if not chans:
                        continue
                    ch = list(chans.values())[0]
                    ch, fs_eff = maybe_downsample(ch, fs)
                    for (a, b) in idx:
                        if not (0 <= a < b <= len(ch)):
                            continue
                        feats = resp_features(ch[a:b], fs_eff)
                        win_rows.append({"t0": float(t[a]), "t1": float(t[b-1]), **feats})

                # free per-file arrays ASAP
                del chans

            # Aggregate per (t0,t1): merge features across modalities
            if not win_rows:
                continue

            dfw = pd.DataFrame(win_rows)
            dfw = dfw.groupby(["t0", "t1"], as_index=False).mean(numeric_only=True)
            dfw = proxy_rules(dfw)

            # Chunked writes to reduce memory pressure
            CHUNK = 500
            with driver.session(database=NEO4J_DATABASE) as s:
                for i in range(0, len(dfw), CHUNK):
                    chunk = dfw.iloc[i:i+CHUNK]
                    for _, r in chunk.iterrows():
                        wid = f"{sid}:{int(r['t0']*1000)}-{int(r['t1']*1000)}"
                        feats = {k: float(v) for k, v in r.items()
                                 if k not in ["t0", "t1", "EEG_anomaly", "Gait_instability", "Stress_flag", "TIA_like"]
                                 and isinstance(v, (int, float, np.floating))}
                        s.execute_write(neo4j_merge_window_features, wid, sid, r["t0"], r["t1"], feats)
                        flags = {
                            "EEG_anomaly": int(r.get("EEG_anomaly", 0)),
                            "Gait_instability": int(r.get("Gait_instability", 0)),
                            "Stress_flag": int(r.get("Stress_flag", 0)),
                            "TIA_like": int(r.get("TIA_like", 0)),
                        }
                        s.execute_write(neo4j_merge_events, wid, flags)
                    del chunk

            # free memory per session
            del win_rows, dfw
            import gc; gc.collect()

    print("✅ KG build completed.")

if __name__ == "__main__":
    print(f"DATA_ROOT = {DATA_ROOT}")
    assert DATA_ROOT.exists(), f"DATA_ROOT does not exist: {DATA_ROOT}"
    build()