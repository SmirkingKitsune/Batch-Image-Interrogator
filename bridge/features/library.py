"""Directory, gallery, inspection and organize operations for the bridge."""

from __future__ import annotations

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from bridge.server import RpcError
from core.file_manager import FileManager
from core.hashing import hash_image_content
from core.pipelines import scan_image_directory
from core.tag_review import (
    apply_common_tag_edits,
    build_tag_comparison,
    collect_editor_tags,
    compute_common_tags,
    extract_wd_ratings,
    plan_apply_to_file,
)

# How many tags the gallery sidebar lists. Laying out thousands of rows does
# not scale, so the most common are shown and search reaches the rest.
MAX_SIDEBAR_TAGS = 200


class LibraryFeature:
    """The shared image source and everything that reads or edits sidecars."""

    def _init_library(self) -> None:
        self.directory: Optional[Path] = None
        self.recursive = False
        self.images: List[str] = []
        self.scanning = False
        self._scan_generation = 0
        self._tag_cache: Dict[str, Tuple[int, int, Tuple[str, ...]]] = {}
        self._tag_cache_lock = threading.Lock()
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._meta_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bridge-meta")
        self._organize_running = False

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    def rpc_dir_open(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        directory = Path(os.path.expanduser(str(path or ""))).resolve()
        if not directory.is_dir():
            raise RpcError(f"Not a directory: {path}", code="not_a_directory")
        if self.interrogation_busy():
            raise RpcError("Wait for the running batch to finish before changing directory.", code="busy")

        with self._lock:
            self.directory = directory
            self.recursive = bool(recursive)
            self._scan_generation += 1
            generation = self._scan_generation
            self.scanning = True
            self.images = []

        # Local-database mode keeps one database per directory.
        self.database.switch_to_directory(str(directory))
        self.ui_settings.set("last_directory", str(directory))
        self.ui_settings.set("recursive", bool(recursive))
        self.bus.publish("dir.scanning", {"path": str(directory), "recursive": bool(recursive)})
        self._spawn("dir-scan", lambda: self._scan(directory, bool(recursive), generation))
        return {"path": str(directory), "recursive": bool(recursive)}

    def rpc_dir_set_recursive(self, recursive: bool) -> Dict[str, Any]:
        if not self.directory:
            with self._lock:
                self.recursive = bool(recursive)
            self.ui_settings.set("recursive", bool(recursive))
            return {"path": "", "recursive": bool(recursive)}
        return self.rpc_dir_open(str(self.directory), recursive)

    def rpc_dir_rescan(self) -> Dict[str, Any]:
        if not self.directory:
            raise RpcError("No directory selected.", code="no_directory")
        return self.rpc_dir_open(str(self.directory), self.recursive)

    def _scan(self, directory: Path, recursive: bool, generation: int) -> None:
        def stale() -> bool:
            return generation != self._scan_generation

        def progress(count: int, current: str) -> None:
            if not stale():
                self.bus.publish("dir.progress", {"count": count, "current": current})

        try:
            paths = scan_image_directory(
                str(directory), recursive=recursive, is_cancelled=stale, on_progress=progress
            )
        except Exception as exc:  # noqa: BLE001
            if not stale():
                with self._lock:
                    self.scanning = False
                self.bus.publish("dir.error", {"message": str(exc)})
            return
        if paths is None or stale():
            return
        with self._lock:
            self.images = paths
            self.scanning = False
        self.bus.publish("dir.loaded", self._dir_payload())
        self._on_images_changed()

    def _dir_payload(self) -> Dict[str, Any]:
        with self._lock:
            images = list(self.images)
            directory = self.directory
            recursive = self.recursive
            scanning = self.scanning
        has_txt = [FileManager.has_text_file(Path(p)) for p in images]
        parents = {os.path.dirname(p) for p in images}
        return {
            "path": str(directory) if directory else "",
            "recursive": recursive,
            "scanning": scanning,
            "paths": images,
            "has_txt": has_txt,
            "dir_count": len(parents),
        }

    def rpc_dir_state(self) -> Dict[str, Any]:
        return self._dir_payload()

    def _on_images_changed(self) -> None:
        """Hook for features that depend on the image list."""
        refresh = getattr(self, "_refresh_batch_sources", None)
        if callable(refresh):
            refresh()

    # ------------------------------------------------------------------
    # Sidecar reading with a stat-keyed cache
    # ------------------------------------------------------------------

    def _read_tags(self, image_path: str) -> Tuple[str, ...]:
        txt_path = FileManager.get_text_file_path(Path(image_path))
        try:
            stat = txt_path.stat()
        except OSError:
            with self._tag_cache_lock:
                self._tag_cache.pop(image_path, None)
            return ()
        with self._tag_cache_lock:
            cached = self._tag_cache.get(image_path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
        tags = tuple(FileManager.read_tags_from_file(Path(image_path)))
        with self._tag_cache_lock:
            self._tag_cache[image_path] = (stat.st_mtime_ns, stat.st_size, tags)
        return tags

    def _write_tags(self, image_path: str, tags: Sequence[str]) -> List[str]:
        FileManager.write_tags_to_file(Path(image_path), list(tags), overwrite=True)
        with self._tag_cache_lock:
            self._tag_cache.pop(image_path, None)
        self._forget_meta(image_path)
        written = list(self._read_tags(image_path))
        self.bus.publish(
            "gallery.tags_saved",
            {"path": image_path, "has_txt": True, "count": len(written)},
        )
        return written

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def rpc_gallery_list(
        self,
        sort: str = "name",
        show: str = "all",
        tags: Optional[List[str]] = None,
        search: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            images = list(self.images)
        required = [tag for tag in (tags or []) if isinstance(tag, str)]

        tag_counts: Counter = Counter()
        rows = []
        for path in images:
            image_tags = self._read_tags(path)
            tag_counts.update(image_tags)
            rows.append((path, image_tags))

        def keep(entry) -> bool:
            path, image_tags = entry
            if required and not all(tag in image_tags for tag in required):
                return False
            has_txt = FileManager.has_text_file(Path(path))
            if show == "tagged" and not has_txt:
                return False
            if show == "untagged" and has_txt:
                return False
            return True

        filtered = [entry for entry in rows if keep(entry)]
        if sort == "date":
            filtered.sort(key=lambda entry: _safe_stat(entry[0], "st_mtime"), reverse=True)
        elif sort == "size":
            filtered.sort(key=lambda entry: _safe_stat(entry[0], "st_size"), reverse=True)
        else:
            filtered.sort(key=lambda entry: Path(entry[0]).name.lower())

        search_text = (search or "").strip().lower()
        selected = sorted(set(required))
        matching = [
            (tag, count)
            for tag, count in tag_counts.items()
            if (not search_text or search_text in tag.lower()) and tag not in required
        ]
        matching.sort(key=lambda item: (-item[1], item[0]))
        room = max(MAX_SIDEBAR_TAGS - len(selected), 0)
        return {
            "items": [
                {"p": path, "t": FileManager.has_text_file(Path(path)), "n": len(image_tags)}
                for path, image_tags in filtered
            ],
            "total": len(images),
            "selected_tags": [[tag, tag_counts.get(tag, 0)] for tag in selected],
            "tags": [[tag, count] for tag, count in matching[:room]],
            "hidden_tags": max(0, len(matching) - room),
            "unique_tags": len(tag_counts),
        }

    def rpc_gallery_meta(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Dimensions and database state for the visible thumbnails."""
        wanted = [p for p in (paths or [])[:200] if isinstance(p, str)]
        return list(self._meta_pool.map(self._image_meta, wanted))

    def _forget_meta(self, path: str) -> None:
        """Drop cached metadata after a new result or sidecar for this image."""
        with self._lock:
            self._meta_cache.pop(path, None)

    def _image_meta(self, path: str) -> Dict[str, Any]:
        with self._lock:
            cached = self._meta_cache.get(path)
        # The sidecar is part of the key: writing it changes has_txt and tags.
        txt_path = str(FileManager.get_text_file_path(Path(path)))
        stat_key = (_safe_stat(path, "st_mtime_ns"), _safe_stat(txt_path, "st_mtime_ns"))
        if cached and cached.get("_stat") == stat_key:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        meta: Dict[str, Any] = {"p": path, "w": None, "h": None, "size": _safe_stat(path, "st_size")}
        try:
            with Image.open(path) as image:
                meta["w"], meta["h"] = image.size
        except Exception:  # noqa: BLE001 - unreadable files still list
            meta["error"] = "decode"
        models: List[str] = []
        try:
            file_hash = hash_image_content(path)
            meta["hash"] = file_hash
            models = [row.get("model_name") for row in self.database.get_all_interrogations_for_image(file_hash) or []]
        except Exception:  # noqa: BLE001
            pass
        meta["models"] = len(models)
        has_txt = FileManager.has_text_file(Path(path))
        meta["t"] = has_txt
        meta["db_only"] = bool(models) and not has_txt
        meta["tags"] = len(self._read_tags(path))
        with self._lock:
            self._meta_cache[path] = {**meta, "_stat": stat_key}
        return meta

    def rpc_gallery_detail(self, path: str) -> Dict[str, Any]:
        image_path = self._require_image(path)
        file_tags = list(self._read_tags(image_path))
        interrogations = self._interrogations_for(image_path)
        all_tags, selected = collect_editor_tags(interrogations, file_tags, self.tag_filters)
        meta = self._image_meta(image_path)
        return {
            "path": image_path,
            "meta": meta,
            "file_tags": file_tags,
            "interrogations": [_slim_interrogation(row) for row in interrogations],
            "editor": {"all": all_tags, "selected": selected},
        }

    def rpc_gallery_save_tags(self, path: str, tags: List[str]) -> Dict[str, Any]:
        image_path = self._require_image(path)
        clean = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        written = self._write_tags(image_path, clean)
        return {"path": image_path, "tags": written}

    def rpc_gallery_common_tags(self, paths: List[str]) -> Dict[str, Any]:
        images = [self._require_image(p) for p in (paths or [])]
        if not images:
            return {"tags": []}

        def per_image():
            for image_path in images:
                rows = self._interrogations_for(image_path)
                yield [row.get("tags", []) for row in rows], list(self._read_tags(image_path))

        common = compute_common_tags(per_image(), self.tag_filters)
        return {"tags": sorted(common, key=str.lower), "count": len(images)}

    def rpc_gallery_save_common(
        self, paths: List[str], original: List[str], selected: List[str]
    ) -> Dict[str, Any]:
        images = [self._require_image(p) for p in (paths or [])]
        original_set = set(original or [])
        selected_set = set(selected or [])
        to_remove = original_set - selected_set
        to_add = selected_set - original_set
        saved, failed = [], []
        for image_path in images:
            try:
                new_tags = apply_common_tag_edits(
                    list(self._read_tags(image_path)), to_remove, to_add, self.tag_filters
                )
                self._write_tags(image_path, new_tags)
                saved.append(image_path)
            except Exception as exc:  # noqa: BLE001
                failed.append({"path": image_path, "error": str(exc)})
        return {"saved": len(saved), "failed": failed, "removed": sorted(to_remove), "added": sorted(to_add)}

    # ------------------------------------------------------------------
    # Advanced inspection
    # ------------------------------------------------------------------

    def rpc_inspect_image(self, path: str) -> Dict[str, Any]:
        image_path = self._require_image(path)
        file_tags = list(self._read_tags(image_path))
        interrogations = self._interrogations_for(image_path)
        meta = self._image_meta(image_path)
        txt_path = FileManager.get_text_file_path(Path(image_path))
        last_write = _safe_stat(str(txt_path), "st_mtime")

        models = []
        for row in interrogations:
            confidence = row.get("confidence_scores") or {}
            comparison = build_tag_comparison(row.get("tags", []), confidence, file_tags, self.tag_filters)
            plan = plan_apply_to_file(row.get("tags", []), confidence, file_tags, self.tag_filters)
            models.append({
                **_slim_interrogation(row),
                "comparison": comparison,
                "plan": {
                    "tags": plan["tags"],
                    "added": plan["added"],
                    "rewritten": [list(pair) for pair in plan["rewritten"]],
                    "kept_manual": plan["kept_manual"],
                    "changes": plan["tags"] != file_tags,
                },
                "ratings": (
                    extract_wd_ratings(row.get("tags", []), confidence)
                    if row.get("model_type") == "WD" else None
                ),
            })

        all_tags, selected = collect_editor_tags(interrogations, file_tags, self.tag_filters)
        prefix = set(self.tag_filters.get_prefix_tags())
        return {
            "path": image_path,
            "meta": meta,
            "file_tags": file_tags,
            "prefix_tags": sorted(prefix),
            "last_write": last_write or None,
            "models": models,
            "editor": {"all": all_tags, "selected": selected},
        }

    def rpc_inspect_apply(self, path: str, model_name: str) -> Dict[str, Any]:
        image_path = self._require_image(path)
        row = next(
            (r for r in self._interrogations_for(image_path) if r.get("model_name") == model_name),
            None,
        )
        if row is None:
            raise RpcError(f"No stored result from {model_name} for this image.")
        plan = plan_apply_to_file(
            row.get("tags", []), row.get("confidence_scores") or {},
            list(self._read_tags(image_path)), self.tag_filters,
        )
        written = self._write_tags(image_path, plan["tags"])
        return {"path": image_path, "tags": written}

    # ------------------------------------------------------------------
    # Organize by tags
    # ------------------------------------------------------------------

    def rpc_organize_scan(self, target: str = "organized") -> Dict[str, Any]:
        root = self._require_directory()
        target = (target or "organized").strip() or "organized"
        counts: Counter = Counter()
        for image in FileManager.find_images(str(root), recursive=True):
            try:
                rel = image.relative_to(root)
            except ValueError:
                continue
            counts["." if len(rel.parts) == 1 else rel.parts[0]] += 1

        subdirs = []
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            name = entry.name
            subdirs.append({
                "rel": name,
                "count": counts.get(name, 0),
                # Folders already holding organized images are not moved again.
                "default_selected": name != target,
            })
        tag_counts: Counter = Counter()
        with self._lock:
            images = list(self.images)
        for image_path in images:
            tag_counts.update(self._read_tags(image_path))
        return {
            "root": str(root),
            "root_count": counts.get(".", 0),
            "subdirs": subdirs,
            "tags": [tag for tag, _count in tag_counts.most_common(400)],
        }

    def rpc_organize_plan(self, **options: Any) -> Dict[str, Any]:
        root = self._require_directory()
        plan = self._organize_plan(root, options)
        return {k: v for k, v in plan.items() if k != "moves"}

    def rpc_organize_run(self, **options: Any) -> Dict[str, Any]:
        root = self._require_directory()
        if self._organize_running:
            raise RpcError("Organize is already running.", code="busy")
        if self.interrogation_busy():
            raise RpcError("Wait for the running batch to finish before moving files.", code="busy")
        plan = self._organize_plan(root, options)
        if not plan["count"]:
            raise RpcError("No images match these tags.")
        self._organize_running = True
        self._spawn("organize", lambda: self._run_organize(root, plan, options))
        return {"count": plan["count"]}

    def _organize_plan(self, root: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        tags = [str(t).strip() for t in options.get("tags") or [] if str(t).strip()]
        match_mode = "all" if options.get("match_mode") == "all" else "any"
        target = (str(options.get("target") or "organized")).strip() or "organized"
        if any(sep in target for sep in ("/", "\\")) or target in (".", ".."):
            raise RpcError("Target subdirectory must be a single folder name.")
        recursive = bool(options.get("recursive", True))
        move_text = bool(options.get("move_text", True))
        target_in_root = bool(options.get("target_in_root", False))
        selected_dirs = set(options.get("selected_dirs") or [])

        if recursive:
            candidates = FileManager.find_images(str(root), recursive=True)
            selected_images = []
            for image in candidates:
                try:
                    rel = image.relative_to(root)
                except ValueError:
                    continue
                source = "." if len(rel.parts) == 1 else rel.parts[0]
                if source in selected_dirs:
                    selected_images.append(image)
            source_count = len(selected_dirs)
        else:
            selected_images = FileManager.find_images(str(root), recursive=False)
            source_count = 1

        wanted = [t.lower() for t in tags]
        moves = []
        txt_count = 0
        sources_used = set()
        planned_names: Dict[Path, set] = {}
        collisions = 0
        if wanted:
            for image in selected_images:
                existing = [t.lower() for t in self._read_tags(str(image))]
                matched = any(t in existing for t in wanted) if match_mode == "any" else all(t in existing for t in wanted)
                if not matched:
                    continue
                destination_dir = (root if target_in_root else image.parent) / target
                names = planned_names.setdefault(destination_dir, set())
                if (destination_dir / image.name).exists() or image.name in names:
                    collisions += 1
                names.add(image.name)
                moves.append(image)
                if move_text and FileManager.has_text_file(image):
                    txt_count += 1
                rel = image.relative_to(root)
                sources_used.add("." if len(rel.parts) == 1 else rel.parts[0])

        destination = str(root / target) if target_in_root else f"<parent>/{target}/"
        return {
            "count": len(moves),
            "txt_count": txt_count,
            "destination": destination,
            "sources_used": len(sources_used),
            "source_count": source_count,
            "collisions": collisions,
            "moves": moves,
            "tags": tags,
            "match_mode": match_mode,
            "target": target,
            "move_text": move_text,
            "target_root": str(root) if target_in_root else None,
        }

    def _run_organize(self, root: Path, plan: Dict[str, Any], options: Dict[str, Any]) -> None:
        moved = 0
        errors = []
        moves = plan["moves"]
        target_root = Path(plan["target_root"]) if plan["target_root"] else None
        try:
            for index, image in enumerate(moves, 1):
                self.bus.publish("organize.progress", {"current": index, "total": len(moves), "name": image.name})
                try:
                    if FileManager.organize_by_tags(
                        image, plan["tags"], plan["target"], plan["move_text"], plan["match_mode"], target_root
                    ):
                        moved += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append({"path": str(image), "error": str(exc)})
        finally:
            self._organize_running = False
            self.bus.publish("organize.finished", {"moved": moved, "errors": errors})
        if self.directory:
            self.rpc_dir_open(str(self.directory), self.recursive)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_directory(self) -> Path:
        if not self.directory:
            raise RpcError("Select a directory first.", code="no_directory")
        return self.directory

    def _require_image(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise RpcError("No image given.")
        target = Path(path)
        if target.suffix.lower() not in FileManager.SUPPORTED_EXTENSIONS or not target.is_file():
            raise RpcError(f"Image not found: {path}", code="not_found")
        return str(target)

    def _interrogations_for(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            return self.database.get_all_interrogations_for_image(hash_image_content(image_path)) or []
        except Exception:  # noqa: BLE001
            return []


def _slim_interrogation(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": row.get("model_name"),
        "model_type": row.get("model_type"),
        "tags": list(row.get("tags") or []),
        "confidence_scores": row.get("confidence_scores") or None,
        "interrogated_at": row.get("interrogated_at"),
    }


def _safe_stat(path: str, field: str):
    try:
        return getattr(os.stat(path), field)
    except OSError:
        return 0
