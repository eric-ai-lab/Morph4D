#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


os.environ["MOJITO_DISABLE_RENDERED_RESULTS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"

# Global batch logging
MASTER_LOG_FILE: Optional[Path] = None
TRACKER_FILE: Optional[Path] = None
TRACKER_LOCK = threading.Lock()


def _append_master_log(line: str) -> None:
    global MASTER_LOG_FILE
    if MASTER_LOG_FILE is None:
        return
    try:
        with MASTER_LOG_FILE.open("a", encoding="utf-8", errors="ignore") as fp:
            fp.write(line.rstrip("\n") + "\n")
    except Exception:
        # avoid crashing on logging failures
        pass


def info(msg: str) -> None:
    print(msg, flush=True)
    _append_master_log(msg)


def write_tracker_record(record: dict) -> None:
    global TRACKER_FILE
    if TRACKER_FILE is None:
        return
    try:
        with TRACKER_LOCK:
            with TRACKER_FILE.open("a", encoding="utf-8", errors="ignore") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], log_file: Path, cwd: Optional[Path] = None, env: Optional[dict] = None) -> Tuple[int, str]:
    ensure_dir(log_file.parent)
    info(f"$ {' '.join(cmd)}")
    with log_file.open("w", encoding="utf-8", errors="ignore") as fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fp.write(line)
            lines.append(line)
            _append_master_log(line)
        proc.wait()
        return proc.returncode, "".join(lines)


def discover_sequences(data_root: Path, sequences: Optional[List[str]]) -> List[Path]:
    if sequences:
        return [data_root / s for s in sequences]
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def ensure_images_subdir(seq_dir: Path) -> None:
    images_dir = seq_dir / "images"
    if images_dir.exists():
        return
    frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if frames:
        images_dir.mkdir(parents=True, exist_ok=True)
        for f in frames:
            shutil.move(str(f), str(images_dir / f.name))


def parse_exp_log_dir_from_output(output: str) -> Optional[str]:
    # run.py prints: EXP_LOG_DIR:<path>
    for line in output.splitlines():
        if line.startswith("EXP_LOG_DIR:"):
            return line.split("EXP_LOG_DIR:", 1)[1].strip()
    return None


def find_latest_exp_log_dir(outputs_dir_for_seq: Path) -> Optional[Path]:
    # Expected structure: outputs/<seq_name>/log/<timestamped_log_dir>
    log_root = outputs_dir_for_seq / "log"
    if not log_root.exists():
        return None
    candidates = [p for p in log_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_exp_log_dir_via_train_logs(seq_name: str) -> Optional[Path]:
    # Look under outputs/logs/<seq_name>/*/05_train.log and parse EXP_LOG_DIR
    logs_root = OUTPUTS_ROOT / "logs" / seq_name
    if not logs_root.exists():
        return None
    # Collect timestamped directories only
    candidates = [p for p in logs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        train_log = cand / "05_train.log"
        if not train_log.exists():
            continue
        try:
            content = train_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        exp_dir_str = parse_exp_log_dir_from_output(content)
        if exp_dir_str:
            exp_path = Path(exp_dir_str)
            if exp_path.exists():
                return exp_path
            # Even if the printed path used a different symlink, still return it
            return exp_path
    return None


@dataclass
class PipelineArgs:
    data_root: Path
    sequences: List[str]
    prepare_config: Path
    train_config: Path
    feature_config: Optional[Path]
    lseg_weights_path: Optional[Path]
    height: int
    width: int
    color_prompt: Optional[str]
    remove_prompt: Optional[str]
    extract_prompt: Optional[str]
    auto_prompts: bool
    num_color_variants: int
    skip_edit: bool
    resume_edit: bool
    resume_stages: bool
    gpu_id: str
    edit_kinds: Optional[List[str]]
    colors: List[str]
    thr_min: Optional[float]
    thr_max: Optional[float]
    gpt_temp: Optional[float]
    num_prompts: Optional[int]
    agent_verbose: bool
    agent_foldername: Optional[str]
    api: str


def run_sequence(seq_name: str, gpu_id: str, pargs: PipelineArgs) -> None:
    seq_dir = pargs.data_root / seq_name
    # Per-sequence timestamped log directory
    ts_seq = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUTS_ROOT / "logs" / seq_name / ts_seq
    ensure_dir(log_dir)
    # Persistent stage marker directory (not timestamped) for resume across runs
    markers_dir = OUTPUTS_ROOT / "logs" / seq_name / "markers"
    ensure_dir(markers_dir)

    info(f"==> [{seq_name}] Using GPU {gpu_id}")
    ensure_images_subdir(seq_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    exp_root: Optional[Path] = None
    start_ts = datetime.utcnow().isoformat()
    stage = "init"

    def _marker(stage_name: str) -> Path:
        return markers_dir / f"{stage_name}.ok"

    def _touch_ok(stage_name: str) -> None:
        try:
            _marker(stage_name).write_text(datetime.utcnow().isoformat())
        except Exception:
            pass

    # Edit outputs helpers (scoped to sequence)
    def _list_tag_videos(exp_root_dir: Optional[Path], tag: str) -> List[Path]:
        if exp_root_dir is None:
            return []
        vdir = exp_root_dir / "editing_output" / tag
        if not vdir.exists():
            return []
        return sorted([p for p in vdir.rglob("*.mp4") if p.is_file()])

    def _write_edit_manifest(stage_tag: str, videos: List[Path]) -> None:
        try:
            (markers_dir / f"{stage_tag}.json").write_text(
                json.dumps([str(p) for p in videos], ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

    def _read_edit_manifest(stage_tag: str) -> List[Path]:
        try:
            mf = markers_dir / f"{stage_tag}.json"
            if mf.exists():
                arr = json.loads(mf.read_text(encoding="utf-8", errors="ignore"))
                return [Path(s) for s in arr if isinstance(s, str)]
        except Exception:
            pass
        return []

    def _move_outputs_to_tag_dir(exp_root_dir: Optional[Path], tag: str, paths: List[Path], stamp: str) -> List[Path]:
        if exp_root_dir is None:
            return paths
        dest_root = exp_root_dir / "editing_output" / tag / stamp
        ensure_dir(dest_root)
        moved: List[Path] = []
        for p in paths:
            try:
                dest = dest_root / p.name
                if dest.exists():
                    base = p.stem
                    ext = p.suffix
                    k = 1
                    # ensure uniqueness within destination
                    while True:
                        candidate = dest_root / f"{base}_{k}{ext}"
                        if not candidate.exists():
                            dest = candidate
                            break
                        k += 1
                shutil.move(str(p), str(dest))
                moved.append(dest)
                # Move sidecar directory if present (same basename without extension)
                sidecar = p.with_suffix("")
                if sidecar.exists() and sidecar.is_dir():
                    sidecar_dest = dest_root / sidecar.name
                    # Avoid collision on sidecar too
                    if sidecar_dest.exists():
                        k2 = 1
                        while True:
                            candidate_dir = dest_root / f"{sidecar.name}_{k2}"
                            if not candidate_dir.exists():
                                sidecar_dest = candidate_dir
                                break
                            k2 += 1
                    shutil.move(str(sidecar), str(sidecar_dest))
            except Exception:
                # If move fails, keep original path
                moved.append(p)
        return moved

    try:
        if not pargs.resume_edit:
            # 1) prepare.py
            stage = "prepare"
            if pargs.resume_stages and _marker("prepare").exists():
                info(f"[{seq_name}] Skipping prepare (resume)")
            else:
                rc, _ = run_cmd(
                    [
                        sys.executable,
                        str(REPO_ROOT / "prepare.py"),
                        "--config",
                        str(pargs.prepare_config),
                        "--src",
                        str(seq_dir),
                    ],
                    log_file=log_dir / "01_prepare.log",
                    cwd=REPO_ROOT,
                    env=env,
                )
                if rc != 0:
                    raise RuntimeError(f"prepare.py failed for {seq_name}")
                _touch_ok("prepare")

            # 2) Feature extraction: InternVideo2
            stage = "internvideo"
            if pargs.resume_stages and _marker("internvideo").exists():
                info(f"[{seq_name}] Skipping internvideo (resume)")
            else:
                rc, _ = run_cmd(
                    [
                        sys.executable,
                        "internvideo_extract_feat.py",
                        "--video_path",
                        str(seq_dir / "preprocess"),
                    ],
                    log_file=log_dir / "02_internvideo.log",
                    cwd=REPO_ROOT / "internvideo_chat_feature",
                    env=env,
                )
                if rc != 0:
                    raise RuntimeError(f"internvideo feature extraction failed for {seq_name}")
                _touch_ok("internvideo")

            # 3) Feature extraction: SAM2
            stage = "sam2"
            if pargs.resume_stages and _marker("sam2").exists():
                info(f"[{seq_name}] Skipping sam2 (resume)")
            else:
                rc, _ = run_cmd(
                    [
                        sys.executable,
                        "sam2_extract_feat.py",
                        "--video_path",
                        str(seq_dir / "preprocess"),
                    ],
                    log_file=log_dir / "03_sam2.log",
                    cwd=REPO_ROOT / "sam2",
                    env=env,
                )
                if rc != 0:
                    raise RuntimeError(f"sam2 feature extraction failed for {seq_name}")
                _touch_ok("sam2")

            # 4) Feature extraction: LSeg
            stage = "lseg"
            lseg_weights_default = REPO_ROOT / "lseg_encoder" / "demo_e200.ckpt"
            lseg_weights_path = pargs.lseg_weights_path or lseg_weights_default
            if not lseg_weights_path.exists():
                raise FileNotFoundError(
                    f"LSeg weights not found at {lseg_weights_path}. Place demo_e200.ckpt in lseg_encoder/."
                )
            if pargs.resume_stages and _marker("lseg").exists():
                info(f"[{seq_name}] Skipping lseg (resume)")
            else:
                rc, _ = run_cmd(
                    [
                        sys.executable,
                        "encode_images.py",
                        "--backbone",
                        "clip_vitl16_384",
                        "--weights",
                        str(lseg_weights_path),
                        "--widehead",
                        "--no-scaleinv",
                        "--outdir",
                        str(seq_dir / "preprocess" / "semantic_features" / "rgb_feature_langseg"),
                        "--test-rgb-dir",
                        str(seq_dir / "preprocess" / "images"),
                        "--workers",
                        "0",
                    ],
                    log_file=log_dir / "04_lseg.log",
                    cwd=REPO_ROOT / "lseg_encoder",
                    env=env,
                )
                if rc != 0:
                    raise RuntimeError(f"lseg feature extraction failed for {seq_name}")
                _touch_ok("lseg")

            # 5) Train / reconstruct
            stage = "train"
            if pargs.resume_stages and _marker("train").exists():
                info(f"[{seq_name}] Skipping train (resume)")
                exp_root = find_latest_exp_log_dir(OUTPUTS_ROOT / seq_name)
                if exp_root is None:
                    # Fallback: parse latest per-sequence 05_train.log
                    exp_root = find_exp_log_dir_via_train_logs(seq_name)
                if exp_root is None:
                    raise RuntimeError(f"--resume-stages set but no EXP_LOG_DIR found under {OUTPUTS_ROOT / seq_name}")
            else:
                train_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "run.py"),
                    "--config",
                    str(pargs.train_config),
                    "--src",
                    str(seq_dir),
                    "--save_dir",
                    str(OUTPUTS_ROOT),
                    "--comment",
                    "batch_edit",
                ]
                if pargs.feature_config:
                    train_cmd.extend(["--feature_config", str(pargs.feature_config)])

                rc, train_output = run_cmd(
                    train_cmd,
                    log_file=log_dir / "05_train.log",
                    cwd=REPO_ROOT,
                    env=env,
                )
                if rc != 0:
                    raise RuntimeError(f"run.py failed for {seq_name}")

                exp_dir_str = parse_exp_log_dir_from_output(train_output)
                if exp_dir_str:
                    exp_root = Path(exp_dir_str)
                else:
                    # Fallback: find latest under outputs/<seq_name>/log
                    exp_root = find_latest_exp_log_dir(OUTPUTS_ROOT / seq_name)
                    if exp_root is None:
                        raise RuntimeError(f"Cannot locate EXP_LOG_DIR for {seq_name}")
                _touch_ok("train")
        else:
            # Resume only editing: discover latest trained EXP_LOG_DIR
            stage = "resume_locate_exp"
            exp_root = find_latest_exp_log_dir(OUTPUTS_ROOT / seq_name)
            if exp_root is None:
                # Fallback: parse latest per-sequence 05_train.log
                exp_root = find_exp_log_dir_via_train_logs(seq_name)
            if exp_root is None:
                raise RuntimeError(
                    f"--resume-edit requested but no EXP_LOG_DIR found under {OUTPUTS_ROOT / seq_name}"
                )

        # 6) Edits (optional)
        if pargs.skip_edit:
            info(f"[{seq_name}] Skipping edits (--skip-edit)")
            write_tracker_record({
                "sequence": seq_name,
                "status": "success",
                "stage": "train_done",
                "exp_log_dir": str(exp_root) if exp_root else None,
                "logs_dir": str(log_dir),
                "gpu_id": gpu_id,
                "start_time": start_ts,
                "end_time": datetime.utcnow().isoformat(),
            })
            return

        # Build dynamic prompts if requested
        def _seq_obj_label(name: str) -> str:
            mapping = {
                "synth_001": "robotic arm",
                "synth_005": "robot car",
                "synth_009": "snowplow",
                "synth_010": "robot archaeologist",
                "synth_011": "bus",
                "synth_013": "jellyfish",
                "synth_018": "exploration rover",
                "synth_019": "robot",
                "synth_020": "robot dog",
                "synth_023": "robotic arm",
                "synth_030": "mars-rover",
                "synth_032": "bus",
                "synth_036": "robot",
                "synth_042": "vehicle",
                "synth_052": "garbage-truck",
                "synth_054": "vehicle",
                "synth_059": "roomba",
                "Robot_Learning_and_Walking_Video": "robot",
                "Robot_Walking_and_Learning_Video": "robot",
            }
            return mapping.get(name, "object")

        def _generate_auto_prompts(name: str) -> Tuple[List[str], str, str]:
            obj = _seq_obj_label(name)
            # Deterministic variety based on sequence name
            palette = [
                f"Make the {obj} purple",
                f"Make the {obj} teal",
                f"Make the {obj} orange",
                f"Make the {obj} red",
                f"Make the {obj} green",
                f"Make the {obj} blue",
                f"Make the {obj} yellow",
            ]
            base = abs(hash(name))
            c1 = palette[base % len(palette)]
            c2 = palette[(base // 7) % len(palette)]
            if c2 == c1:
                c2 = palette[(base // 5 + 1) % len(palette)]
            # Also include deletion/extraction prompts for agentic runs
            remove_p = f"delete the {obj}"
            extract_p = f"extract the {obj}"
            return [c1, c2][: max(1, pargs.num_color_variants)], remove_p, extract_p

        auto_color_prompts: List[str] = []
        auto_remove_prompt: Optional[str] = None
        auto_extract_prompt: Optional[str] = None
        if pargs.auto_prompts:
            auto_color_prompts, auto_remove_prompt, auto_extract_prompt = _generate_auto_prompts(seq_name)

        # Resolve final prompts honoring manual overrides when provided
        final_color_prompts: List[str]
        # If explicit colors are provided, synthesize prompts with object-aware phrasing
        explicit_colors: List[str] = getattr(pargs, "colors", []) or []
        if explicit_colors:
            obj_label = _seq_obj_label(seq_name)
            final_color_prompts = [f"Make the {obj_label} {c}" for c in explicit_colors]
        elif pargs.color_prompt:
            # Manual single color prompt
            final_color_prompts = [pargs.color_prompt]
        else:
            final_color_prompts = auto_color_prompts
        # Include remove/extract if auto prompts are enabled; allow CLI to override
        final_remove_prompt = pargs.remove_prompt or auto_remove_prompt
        final_extract_prompt = pargs.extract_prompt or auto_extract_prompt

        # Filter by selected edit kinds when provided
        selected_kinds: Optional[set] = None
        if pargs.edit_kinds is not None:
            try:
                ekinds = pargs.edit_kinds or []
                selected_kinds = set([k.strip().lower() for k in ekinds if isinstance(k, str) and k.strip()])
            except Exception:
                selected_kinds = None
        if selected_kinds is not None:
            if "color" not in selected_kinds:
                final_color_prompts = []
            if "remove" not in selected_kinds:
                final_remove_prompt = None
            if "extract" not in selected_kinds:
                final_extract_prompt = None

        # Helper to record prompts near videos and collect new videos
        def _list_videos(vdir: Path) -> List[Path]:
            if not vdir.exists():
                return []
            return sorted([p for p in vdir.rglob("*.mp4") if p.is_file()])

        def _write_prompts_files(prompts_dir: Path, contents: List[Tuple[str, Optional[str]]]) -> None:
            ensure_dir(prompts_dir)
            manifest = {"edits": []}
            for tag, text in contents:
                if text:
                    try:
                        (prompts_dir / f"{tag}.txt").write_text(text)
                    except Exception:
                        pass
                manifest["edits"].append({"tag": tag, "prompt": text})
            try:
                (prompts_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            except Exception:
                pass

        def _load_saved_prompts(prompts_dir: Path) -> Tuple[List[str], Optional[str], Optional[str]]:
            colors: List[str] = []
            remove_p: Optional[str] = None
            extract_p: Optional[str] = None
            try:
                man = prompts_dir / "manifest.json"
                if man.exists():
                    data = json.loads(man.read_text(encoding="utf-8", errors="ignore"))
                    for entry in data.get("edits", []):
                        tag = str(entry.get("tag", ""))
                        prompt = entry.get("prompt")
                        if not prompt:
                            continue
                        if tag.startswith("color"):
                            colors.append(prompt)
                        elif tag == "remove":
                            remove_p = prompt
                        elif tag == "extract":
                            extract_p = prompt
                else:
                    # Fallback to individual files
                    for p in sorted(prompts_dir.glob("color*.txt")):
                        try:
                            colors.append(p.read_text(encoding="utf-8", errors="ignore").strip())
                        except Exception:
                            continue
                    rp = prompts_dir / "remove.txt"
                    if rp.exists():
                        try:
                            remove_p = rp.read_text(encoding="utf-8", errors="ignore").strip()
                        except Exception:
                            pass
                    ep = prompts_dir / "extract.txt"
                    if ep.exists():
                        try:
                            extract_p = ep.read_text(encoding="utf-8", errors="ignore").strip()
                        except Exception:
                            pass
            except Exception:
                # On any error, return whatever we gathered
                pass
            # Sanitize empties
            colors = [c for c in colors if isinstance(c, str) and c.strip()]
            remove_p = remove_p if (isinstance(remove_p, str) and remove_p.strip()) else None
            extract_p = extract_p if (isinstance(extract_p, str) and extract_p.strip()) else None
            return colors, remove_p, extract_p

        def run_edit(prompt: Optional[str], tag: str, idx: int) -> List[Path]:
            if not prompt:
                info(f"[{seq_name}] Skipping {tag} edit (no prompt)")
                return []
            nonlocal stage
            stage = f"edit_{tag}"
            # Resume: if marker exists, return previously recorded videos
            force_redo = pargs.edit_kinds is not None
            if pargs.resume_stages and not force_redo and _marker(stage).exists():
                info(f"[{seq_name}] Skipping {tag} edit (resume)")
                vids = _read_edit_manifest(stage)
                if vids:
                    return vids
                return _list_tag_videos(exp_root, tag)
            editing_output_dir = Path(str(exp_root)) / "editing_output" if exp_root else None
            # If forcing redo, ignore previous outputs so that all current outputs are treated as new
            if editing_output_dir and not force_redo:
                before = set(_list_videos(editing_output_dir))
            else:
                before = set()
            rc, _ = run_cmd(
                [
                    sys.executable,
                    str(REPO_ROOT / "viz_agent.py"),
                    "--config",
                    str(pargs.train_config),
                    "--root",
                    str(exp_root),
                    "--user_prompt",
                    prompt,
                    "--H",
                    str(pargs.height),
                    "--W",
                    str(pargs.width),
                    "--output_root",
                    str(OUTPUTS_ROOT),
                    # passthrough agent tuning
                    *( ["--thr_min", str(pargs.thr_min)] if pargs.thr_min is not None else []),
                    *( ["--thr_max", str(pargs.thr_max)] if pargs.thr_max is not None else []),
                    *( ["--gpt_temp", str(pargs.gpt_temp)] if pargs.gpt_temp is not None else []),
                    *( ["--num_prompt", str(pargs.num_prompts)] if pargs.num_prompts is not None else []),
                    *( ["--foldername", str(pargs.agent_foldername)] if pargs.agent_foldername else []),
                    *( ["--verbose"] if pargs.agent_verbose else []),
                    "--api", str(pargs.api),
                ],
                log_file=log_dir / f"0{6+idx}_edit_{tag}.log",
                cwd=REPO_ROOT,
                env=env,
            )
            if rc != 0:
                raise RuntimeError(f"viz_agent failed for {seq_name} ({tag})")
            after = set(_list_videos(editing_output_dir)) if editing_output_dir else set()
            new_paths = sorted(list(after - before))
            # Move new outputs into per-tag directory to avoid overwrites
            stamp = f"{ts_seq}_idx{idx}"
            moved_paths = _move_outputs_to_tag_dir(exp_root, tag, new_paths, stamp)
            # Mark completion and persist manifest for resume
            _touch_ok(stage)
            _write_edit_manifest(stage, moved_paths)
            return moved_paths

        # Prepare prompts directory near videos
        prompts_dir: Optional[Path] = None
        if exp_root is not None:
            prompts_dir = exp_root / "editing_output" / "prompts"
            try:
                ensure_dir(prompts_dir)
            except Exception:
                pass

        # If resuming edits and specific kinds are requested but no prompts were provided,
        # attempt to reuse previously saved prompts so that edits are re-executed.
        if pargs.resume_edit and pargs.edit_kinds is not None and prompts_dir is not None:
            no_color = len(final_color_prompts) == 0
            no_remove = final_remove_prompt is None
            no_extract = final_extract_prompt is None
            if no_color and no_remove and no_extract:
                saved_colors, saved_remove, saved_extract = _load_saved_prompts(prompts_dir)
                # Fill from saved prompts
                final_color_prompts = saved_colors or []
                final_remove_prompt = saved_remove
                final_extract_prompt = saved_extract
                # Re-apply edit kind filtering just in case
                if selected_kinds is not None:
                    if "color" not in selected_kinds:
                        final_color_prompts = []
                    if "remove" not in selected_kinds:
                        final_remove_prompt = None
                    if "extract" not in selected_kinds:
                        final_extract_prompt = None
                # If still nothing found, fall back to deterministic auto prompts
                if (len(final_color_prompts) == 0) and (final_remove_prompt is None) and (final_extract_prompt is None):
                    auto_colors, auto_remove, auto_extract = _generate_auto_prompts(seq_name)
                    final_color_prompts = auto_colors
                    final_remove_prompt = auto_remove
                    final_extract_prompt = auto_extract
                    if selected_kinds is not None:
                        if "color" not in selected_kinds:
                            final_color_prompts = []
                        if "remove" not in selected_kinds:
                            final_remove_prompt = None
                        if "extract" not in selected_kinds:
                            final_extract_prompt = None

        # Write prompt text files up front
        if prompts_dir is not None:
            prompt_entries: List[Tuple[str, Optional[str]]] = []
            # up to two color variants
            for c_idx, cp in enumerate(final_color_prompts):
                prompt_entries.append((f"color{c_idx+1}", cp))
            # Only write remove/extract if explicitly provided
            if final_remove_prompt:
                prompt_entries.append(("remove", final_remove_prompt))
            if final_extract_prompt:
                prompt_entries.append(("extract", final_extract_prompt))
            _write_prompts_files(prompts_dir, prompt_entries)

        # Execute edits following requested priority. Default: color -> remove -> extract
        collected_videos: List[Tuple[str, List[Path]]] = []
        edit_idx = 0
        exec_order = pargs.edit_kinds if pargs.edit_kinds is not None else ["color", "remove", "extract"]
        plan: List[Tuple[str, Optional[str]]] = []
        for kind in exec_order:
            if kind == "color":
                for c_idx, cp in enumerate(final_color_prompts):
                    plan.append((f"color{c_idx+1}", cp))
            elif kind == "remove":
                if final_remove_prompt:
                    plan.append(("remove", final_remove_prompt))
            elif kind == "extract":
                if final_extract_prompt:
                    plan.append(("extract", final_extract_prompt))
        for tag, prompt in plan:
            vids = run_edit(prompt, tag, edit_idx)
            collected_videos.append((tag, vids))
            edit_idx += 1

        # Summarize outputs and write manifest near videos
        outputs_summary = {
            "sequence": seq_name,
            "exp_log_dir": str(exp_root) if exp_root else None,
            "videos_dir": str(exp_root / "editing_output") if exp_root else None,
            "edits": [
                {"tag": tag, "videos": [str(p) for p in vids]} for tag, vids in collected_videos
            ],
        }
        if prompts_dir is not None:
            try:
                (prompts_dir / "videos_manifest.json").write_text(json.dumps(outputs_summary, ensure_ascii=False, indent=2))
            except Exception:
                pass
        # Print the final video paths to master log
        for tag, vids in collected_videos:
            if vids:
                info(f"[{seq_name}] {tag} videos: ")
                for vp in vids:
                    info(f"  - {vp}")
            else:
                info(f"[{seq_name}] {tag} produced no new videos")

        write_tracker_record({
            "sequence": seq_name,
            "status": "success",
            "stage": "done",
            "exp_log_dir": str(exp_root) if exp_root else None,
            "logs_dir": str(log_dir),
            "gpu_id": gpu_id,
            "start_time": start_ts,
            "end_time": datetime.utcnow().isoformat(),
            "videos": outputs_summary.get("edits", []),
        })
    except Exception as e:
        write_tracker_record({
            "sequence": seq_name,
            "status": "failed",
            "stage": stage,
            "error": str(e),
            "exp_log_dir": str(exp_root) if exp_root else None,
            "logs_dir": str(log_dir),
            "gpu_id": gpu_id,
            "start_time": start_ts,
            "end_time": datetime.utcnow().isoformat(),
        })
        info(f"[{seq_name}] FAILED at stage={stage}: {e}")
        # Do not re-raise; keep batch running
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch edit orchestrator (Python)")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sequences", type=str, default="", help="space-delimited list")
    parser.add_argument("--ngpus", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default="", help=",-separated ids")
    parser.add_argument("--prepare-config", type=Path, default=REPO_ROOT / "configs/wild/prepare_davis.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "configs/wild/davis.yaml")
    parser.add_argument("--feature-config", type=Path, default=None)
    parser.add_argument("--lseg-weights-path", type=Path, default=None)
    parser.add_argument("--color-prompt", type=str, default="")
    parser.add_argument("--remove-prompt", type=str, default="")
    parser.add_argument("--extract-prompt", type=str, default="")
    parser.add_argument("--edit-kinds", type=str, default="", help="comma/space-separated subset of: color,remove,extract")
    parser.add_argument("--colors", type=str, default="", help="comma/space-separated color names for color edits")
    # Agent tuning passthroughs
    parser.add_argument("--thr-min", "--thr_min", type=float, default=None, help="lower bound for agent threshold schedule")
    parser.add_argument("--thr-max", "--thr_max", type=float, default=None, help="upper bound for agent threshold schedule")
    parser.add_argument("--gpt-temp", type=float, default=None, help="temperature for agent LLM sampling")
    parser.add_argument("--num-prompts", type=int, default=None, help="number of prompts agent will generate")
    parser.add_argument("--agent-verbose", action="store_true", help="enable verbose flag inside viz_agent")
    parser.add_argument("--agent-foldername", type=str, default=None, help="subfolder name used by viz_agent")
    parser.add_argument("--api", type=str, default="openrouter", help="LLM API provider: openrouter | xh-gpt4.1")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--batch-log-dir", type=Path, default=None, help="optional custom batch log directory")
    parser.add_argument("--skip-edit", action="store_true", help="run until training, skip viz_agent")
    parser.add_argument("--resume-edit", action="store_true", help="run only viz_agent using latest EXP_LOG_DIR")
    parser.add_argument("--resume-stages", action="store_true", help="resume pipeline stages per sequence using markers")
    parser.add_argument("--auto-prompts", action="store_true", help="auto-generate prompts per sequence object")
    parser.add_argument("--num-color-variants", type=int, default=2, help="number of color edits when auto prompts are enabled")
    args = parser.parse_args()

    sequences = [s for s in args.sequences.split() if s] if args.sequences else []
    seq_paths = discover_sequences(args.data_root, sequences)
    if not seq_paths:
        raise SystemExit(f"No sequences found under {args.data_root}")

    # GPU ids resolution
    gpu_id_arr: List[str]
    if args.gpu_ids:
        gpu_id_arr = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    elif args.ngpus and args.ngpus > 0:
        gpu_id_arr = [str(i) for i in range(args.ngpus)]
    else:
        gpu_id_arr = ["0"]
    info(f"Using {len(gpu_id_arr)} GPU(s): {' '.join(gpu_id_arr)}")

    ensure_dir(OUTPUTS_ROOT / "logs")

    # Setup batch logging directory
    if args.batch_log_dir:
        batch_log_dir = args.batch_log_dir
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_log_dir = OUTPUTS_ROOT / "batch_logs" / ts
    ensure_dir(batch_log_dir)

    global MASTER_LOG_FILE, TRACKER_FILE
    MASTER_LOG_FILE = batch_log_dir / "batch.log"
    TRACKER_FILE = batch_log_dir / "tracker.jsonl"
    info(f"Batch logs: {batch_log_dir}")

    # Prepare arg container
    # Parse edit kinds and colors from CLI
    def _parse_list(s: str) -> List[str]:
        if not s:
            return []
        # allow both comma and space separated
        tokens: List[str] = []
        for part in s.replace("\n", " ").replace(",", " ").split():
            t = part.strip()
            if t:
                tokens.append(t)
        return tokens

    raw_kinds = [k.lower() for k in _parse_list(getattr(args, "edit_kinds", ""))]
    # normalize synonyms
    alias_map = {
        "delete": "remove",
        "deletion": "remove",
        "remove": "remove",
        "color": "color",
        "colour": "color",
        "extract": "extract",
    }
    normalized = [alias_map.get(k, "") for k in raw_kinds]
    # validate values but do not hard error; silently ignore unknown kinds
    valid_kinds = {"color", "remove", "extract"}
    edit_kinds_list = [k for k in normalized if k in valid_kinds]
    colors_list = _parse_list(getattr(args, "colors", ""))

    # map CLI names (kebab) to internal names (snake)
    thr_min = getattr(args, "thr_min", None)
    if thr_min is None:
        thr_min = getattr(args, "thr-min", None)
    thr_max = getattr(args, "thr_max", None)
    if thr_max is None:
        thr_max = getattr(args, "thr-max", None)
    gpt_temp = getattr(args, "gpt_temp", None)
    if gpt_temp is None:
        gpt_temp = getattr(args, "gpt-temp", None)
    num_prompts = getattr(args, "num_prompts", None)
    if num_prompts is None:
        num_prompts = getattr(args, "num-prompts", None)
    agent_foldername = getattr(args, "agent_foldername", None)
    if agent_foldername is None:
        agent_foldername = getattr(args, "agent-foldername", None)

    pargs = PipelineArgs(
        data_root=args.data_root,
        sequences=[p.name for p in seq_paths],
        prepare_config=args.prepare_config,
        train_config=args.train_config,
        feature_config=args.feature_config,
        lseg_weights_path=args.lseg_weights_path,
        height=args.height,
        width=args.width,
        color_prompt=args.color_prompt or None,
        remove_prompt=args.remove_prompt or None,
        extract_prompt=args.extract_prompt or None,
        auto_prompts=bool(args.auto_prompts),
        num_color_variants=int(args.num_color_variants),
        skip_edit=bool(args.skip_edit),
        resume_edit=bool(args.resume_edit),
        resume_stages=bool(args.resume_stages),
        gpu_id="0",  # placeholder; per-sequence assignment below
        edit_kinds=edit_kinds_list or None,
        colors=colors_list,
        thr_min=thr_min,
        thr_max=thr_max,
        gpt_temp=gpt_temp,
        num_prompts=num_prompts,
        agent_verbose=bool(getattr(args, "agent_verbose", False)),
        agent_foldername=agent_foldername,
        api=str(getattr(args, "api", "openrouter")),
    )

    # Round-robin dispatch limited by number of GPUs
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_id_arr)) as ex:
        for idx, seq_path in enumerate(seq_paths):
            gpu_id = gpu_id_arr[idx % len(gpu_id_arr)]
            fut = ex.submit(
                run_sequence,
                seq_path.name,
                gpu_id,
                pargs,
            )
            futures.append(fut)
        # Raise any exceptions
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                # Should not happen because worker handles exceptions, but guard anyway
                info(f"Worker raised exception: {e}")

    info(f"All sequences processed. Outputs are under: {OUTPUTS_ROOT}")


if __name__ == "__main__":
    main()


