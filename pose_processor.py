"""Utilities for generating OpenPose-based representations for VLM inputs."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


class PoseProcessingError(RuntimeError):
    """Raised when OpenPose processing fails or cannot be executed."""


@dataclass(slots=True)
class PoseRepresentation:
    """Container for pose outputs consumed by the prompting pipeline."""

    base64_image: str
    mime_type: str
    summary_text: str
    keypoints: Sequence[Tuple[float, float, float]]
    metadata: Dict[str, object]


class OpenPoseProcessor:
    """Runs OpenPose and converts the results into a VLM-friendly format."""

    # Body25 indexing as documented by the OpenPose project
    BODY25_NAMES: Sequence[str] = (
        "Nose",
        "Neck",
        "Right Shoulder",
        "Right Elbow",
        "Right Wrist",
        "Left Shoulder",
        "Left Elbow",
        "Left Wrist",
        "Mid Hip",
        "Right Hip",
        "Right Knee",
        "Right Ankle",
        "Left Hip",
        "Left Knee",
        "Left Ankle",
        "Right Eye",
        "Left Eye",
        "Right Ear",
        "Left Ear",
        "Left Big Toe",
        "Left Small Toe",
        "Left Heel",
        "Right Big Toe",
        "Right Small Toe",
        "Right Heel",
    )

    BODY25_CONNECTIONS: Sequence[Tuple[int, int]] = (
        (0, 1),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8),
        (8, 9), (9, 10), (10, 11),
        (8, 12), (12, 13), (13, 14),
        (0, 15), (15, 17),
        (0, 16), (16, 18),
        (14, 19), (19, 20), (20, 21),
        (11, 22), (22, 23), (23, 24),
        (11, 24), (14, 21)
    )

    LEFT_INDECES = {5, 6, 7, 12, 13, 14, 19, 20, 21}
    RIGHT_INDECES = {2, 3, 4, 9, 10, 11, 22, 23, 24}

    SUMMARY_KEYPOINTS: Sequence[Tuple[str, int]] = (
        ("Head / Neck", 1),
        ("Torso center", 8),
        ("Right wrist", 4),
        ("Left wrist", 7),
        ("Right knee", 10),
        ("Left knee", 13),
        ("Right ankle", 11),
        ("Left ankle", 14),
    )

    def __init__(self, config: Dict[str, object], dataset_root: Path):
        self.enabled = bool(config.get("enabled", True))
        if not self.enabled:
            raise PoseProcessingError("OpenPose processor disabled in configuration.")

        binary_path = str(config.get("binary_path") or "").strip()
        if not binary_path:
            binary_path = shutil.which("openpose.bin") or shutil.which("OpenPoseDemo.app") or shutil.which("openpose")
        if not binary_path:
            raise PoseProcessingError(
                "OpenPose binary not configured. Set openpose.binary_path or ensure 'openpose.bin' is on PATH."
            )
        self.binary_path = Path(binary_path)
        if not self.binary_path.exists():
            resolved = shutil.which(str(self.binary_path))
            if resolved:
                self.binary_path = Path(resolved)

        self.model_folder = config.get("model_folder") or None
        self.net_resolution = config.get("net_resolution") or None
        self.dynamic_batching = bool(config.get("dynamic_batching", False))
        self.reuse_cache = bool(config.get("reuse_cache", True))
        self.min_confidence = float(config.get("min_confidence", 0.2))
        self.strict = bool(config.get("strict", True))
        self.dataset_root = Path(dataset_root).resolve()

        output_dir = Path(config.get("output_dir", "pose_cache"))
        self.json_dir = output_dir.joinpath("json")
        self.render_dir = output_dir.joinpath("renders")
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, image_path: Path) -> PoseRepresentation:
        """Create an OpenPose-based representation for the provided image."""

        image_path = image_path.resolve()

        relative_path = self._relative_to_dataset(image_path)
        relative_dir = relative_path.parent if relative_path.parent != Path('.') else Path()
        stem = relative_path.stem
        json_path = self.json_dir.joinpath(relative_dir, f"{stem}_keypoints.json")
        render_path = self.render_dir.joinpath(relative_dir, f"{stem}_pose.png")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        render_path.parent.mkdir(parents=True, exist_ok=True)

        if not (self.reuse_cache and json_path.exists()):
            self._run_openpose(image_path, json_path.parent)

        keypoints = self._load_keypoints(json_path)
        skeleton_image = self._draw_pose(image_path, keypoints)
        skeleton_image.save(render_path)
        summary_text = self._summarize_pose(keypoints, skeleton_image.size)

        buffer = io.BytesIO()
        skeleton_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        metadata = {
            "json_path": str(json_path),
            "render_path": str(render_path),
            "relative_image_path": str(relative_path),
            "keypoints_detected": len([kp for kp in keypoints if kp[2] >= self.min_confidence]),
        }

        return PoseRepresentation(
            base64_image=base64_image,
            mime_type="image/png",
            summary_text=summary_text,
            keypoints=keypoints,
            metadata=metadata,
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _relative_to_dataset(self, image_path: Path) -> Path:
        try:
            return image_path.relative_to(self.dataset_root)
        except ValueError:
            return Path(image_path.name)

    def _run_openpose(self, image_path: Path, output_dir: Path) -> None:
        if not self.binary_path.exists() and not shutil.which(self.binary_path.name):
            message = f"OpenPose binary not found at {self.binary_path}."
            if self.strict:
                raise PoseProcessingError(message)
            raise PoseProcessingError(message + " Set strict=false to ignore.")

        cmd: List[str] = [str(self.binary_path), "--image_path", str(image_path), "--write_json", str(output_dir), "--display", "0", "--render_pose", "0"]

        if self.model_folder:
            cmd.extend(["--model_folder", str(self.model_folder)])
        if self.net_resolution:
            cmd.extend(["--net_resolution", str(self.net_resolution)])
        if self.dynamic_batching:
            cmd.append("--dynamic_batching")

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            if self.strict:
                raise PoseProcessingError(
                    "OpenPose command failed with code {}: {}".format(process.returncode, process.stderr.strip())
                )

    def _load_keypoints(self, json_path: Path) -> List[Tuple[float, float, float]]:
        if not json_path.exists():
            if self.strict:
                raise PoseProcessingError(f"OpenPose output missing: {json_path}")
            return []

        with json_path.open("r") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            return []

        pose_data = people[0].get("pose_keypoints_2d", [])
        keypoints = []
        for i in range(0, len(pose_data), 3):
            x, y, c = pose_data[i:i+3]
            keypoints.append((float(x), float(y), float(c)))
        return keypoints

    def _draw_pose(self, image_path: Path, keypoints: Sequence[Tuple[float, float, float]]) -> Image.Image:
        with Image.open(image_path) as original:
            width, height = original.size
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw skeleton edges
        for idx1, idx2 in self.BODY25_CONNECTIONS:
            p1 = self._get_point(keypoints, idx1)
            p2 = self._get_point(keypoints, idx2)
            if not p1 or not p2:
                continue

            color = self._edge_color(idx1, idx2)
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=color, width=4)

        # Draw keypoints as circles
        radius = 6
        for idx, point in enumerate(keypoints):
            x, y, c = point
            if c < self.min_confidence:
                continue
            color = self._point_color(idx)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        return canvas

    def _summarize_pose(self, keypoints: Sequence[Tuple[float, float, float]], image_size: Tuple[int, int]) -> str:
        width, height = image_size
        if not keypoints:
            return "OpenPose detected no person; treat the pose image as blank."

        lines: List[str] = []
        for label, idx in self.SUMMARY_KEYPOINTS:
            point = self._get_point(keypoints, idx)
            if not point:
                lines.append(f"{label}: not confidently detected")
                continue
            x, y, c = point
            x_norm = x / width if width else 0.0
            y_norm = y / height if height else 0.0
            horizontal = self._bucket_axis(x_norm, ("left", "center", "right"))
            vertical = self._bucket_axis(y_norm, ("upper", "mid", "lower"))
            lines.append(
                f"{label}: {horizontal}-{vertical} quadrant (confidence {c:.2f})"
            )

        active_limbs = sum(1 for kp in keypoints if kp and kp[2] >= self.min_confidence)
        lines.append(f"Detected joints over confidence threshold: {active_limbs}")
        return "\n".join(lines)

    def _get_point(self, keypoints: Sequence[Tuple[float, float, float]], index: int) -> Optional[Tuple[float, float, float]]:
        if index >= len(keypoints):
            return None
        x, y, c = keypoints[index]
        if c < self.min_confidence:
            return None
        return x, y, c

    def _edge_color(self, idx1: int, idx2: int) -> Tuple[int, int, int]:
        if idx1 in self.LEFT_INDECES or idx2 in self.LEFT_INDECES:
            return 0, 102, 204  # blue-ish
        if idx1 in self.RIGHT_INDECES or idx2 in self.RIGHT_INDECES:
            return 204, 51, 51  # red-ish
        return 0, 0, 0

    def _point_color(self, idx: int) -> Tuple[int, int, int]:
        if idx in self.LEFT_INDECES:
            return 30, 136, 229  # left side
        if idx in self.RIGHT_INDECES:
            return 229, 57, 53  # right side
        return 55, 71, 79  # central points

    @staticmethod
    def _bucket_axis(value: float, labels: Iterable[str]) -> str:
        segments = list(labels)
        if not segments:
            return "unknown"
        value = max(0.0, min(0.9999, value))
        step = 1.0 / len(segments)
        idx = min(int(value / step), len(segments) - 1)
        return segments[idx]


