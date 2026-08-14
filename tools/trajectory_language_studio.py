#!/usr/bin/env python3
"""Local window for generating language annotations on approved trajectories."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_enrichment_command(
    dataset: Path,
    annotator_spec: str,
    quality_report: Path | None = None,
    out_dir: Path | None = None,
    approved_episodes: Path | None = None,
    profile_dir: Path | None = None,
    annotator_config: Path | None = None,
    overwrite: bool = False,
    python_executable: str = sys.executable,
) -> list[str]:
    if not annotator_spec.strip():
        raise ValueError("language annotator module:attribute is required")
    command = [
        python_executable,
        str(REPO_ROOT / "pipeline" / "trajectory_language_enrichment.py"),
        str(dataset),
        "--annotator",
        annotator_spec.strip(),
    ]
    for flag, path in (
        ("--quality-report", quality_report),
        ("--out-dir", out_dir),
        ("--approved-episodes", approved_episodes),
        ("--task-profile-dir", profile_dir),
        ("--annotator-config", annotator_config),
    ):
        if path is not None:
            command.extend([flag, str(path)])
    if overwrite:
        command.append("--overwrite")
    return command


class TrajectoryLanguageStudio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Trajectory Language Studio")
        self.root.geometry("1120x650")
        self.root.minsize(900, 560)

        self.dataset = tk.StringVar()
        self.quality_report = tk.StringVar()
        self.approved_episodes = tk.StringVar()
        self.profile_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.annotator_spec = tk.StringVar()
        self.annotator_config = tk.StringVar()
        self.overwrite = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.run_button: ttk.Button | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(10, weight=1)

        ttk.Label(
            root_frame,
            text=(
                "By default only keep episodes are annotated. An explicit approved list may "
                "include manually accepted review episodes; drop episodes are always rejected."
            ),
            wraplength=1020,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self._path_row(root_frame, 1, "Dataset", self.dataset, self._browse_dataset)
        self._path_row(
            root_frame,
            2,
            "Quality report (optional default)",
            self.quality_report,
            self._browse_quality_report,
        )
        self._path_row(
            root_frame,
            3,
            "Approved episodes (optional)",
            self.approved_episodes,
            self._browse_approved_episodes,
        )
        self._path_row(
            root_frame,
            4,
            "Task Profile directory (optional)",
            self.profile_dir,
            self._browse_profile_dir,
        )
        self._path_row(
            root_frame,
            5,
            "Output directory (optional default)",
            self.out_dir,
            self._browse_out_dir,
        )
        ttk.Label(root_frame, text="Annotator module:attribute").grid(row=6, column=0, sticky="w")
        ttk.Entry(root_frame, textvariable=self.annotator_spec).grid(
            row=6, column=1, sticky="ew", padx=8, pady=3
        )
        ttk.Label(root_frame, text="required").grid(row=6, column=2, sticky="w")
        self._path_row(
            root_frame,
            7,
            "Annotator config JSON (optional)",
            self.annotator_config,
            self._browse_annotator_config,
        )

        actions = ttk.Frame(root_frame)
        actions.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        ttk.Checkbutton(actions, text="Overwrite existing output", variable=self.overwrite).pack(
            side=tk.LEFT
        )
        self.run_button = ttk.Button(actions, text="Generate annotations", command=self.run)
        self.run_button.pack(side=tk.RIGHT)

        ttk.Label(
            root_frame,
            text="Annotations are written as external sidecars; the source dataset is never modified.",
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(0, 6))

        log_frame = ttk.LabelFrame(root_frame, text="Output", padding=6)
        log_frame.grid(row=10, column=0, columnspan=3, sticky="nsew")
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(root_frame, textvariable=self.status).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(5, 0)
        )

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=3
        )
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2)

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.dataset.set(path)

    def _browse_quality_report(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Quality report", "quality_report.csv"), ("CSV", "*.csv")]
        )
        if path:
            self.quality_report.set(path)

    def _browse_approved_episodes(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Episode list", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.approved_episodes.set(path)

    def _browse_profile_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.profile_dir.set(path)

    def _browse_out_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_dir.set(path)

    def _browse_annotator_config(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.annotator_config.set(path)

    @staticmethod
    def _optional_path(value: str) -> Path | None:
        return Path(value).expanduser() if value.strip() else None

    def run(self) -> None:
        try:
            dataset = Path(self.dataset.get()).expanduser()
            quality_report = self._optional_path(self.quality_report.get())
            approved = self._optional_path(self.approved_episodes.get())
            profile_dir = self._optional_path(self.profile_dir.get())
            out_dir = self._optional_path(self.out_dir.get())
            annotator_config = self._optional_path(self.annotator_config.get())
            if not dataset.is_dir():
                raise FileNotFoundError(f"dataset directory not found: {dataset}")
            for label, path in (
                ("quality report", quality_report),
                ("approved episode list", approved),
                ("annotator config", annotator_config),
            ):
                if path is not None and not path.is_file():
                    raise FileNotFoundError(f"{label} not found: {path}")
            if profile_dir is not None and not profile_dir.is_dir():
                raise FileNotFoundError(f"Task Profile directory not found: {profile_dir}")
            command = build_enrichment_command(
                dataset=dataset,
                annotator_spec=self.annotator_spec.get(),
                quality_report=quality_report,
                out_dir=out_dir,
                approved_episodes=approved,
                profile_dir=profile_dir,
                annotator_config=annotator_config,
                overwrite=self.overwrite.get(),
            )
        except Exception as exc:
            messagebox.showerror("Cannot start language enrichment", str(exc))
            return

        if self.run_button is not None:
            self.run_button.state(["disabled"])
        self.status.set("Language enrichment running...")
        self._append_log("\n$ " + subprocess.list2cmdline(command) + "\n")
        threading.Thread(target=self._run_process, args=(command,), daemon=True).start()

    def _run_process(self, command: list[str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.root.after(0, self._append_log, line)
            return_code = process.wait()
            self.root.after(0, self._finished, return_code)
        except Exception as exc:
            self.root.after(0, self._failed, str(exc))

    def _finished(self, return_code: int) -> None:
        if self.run_button is not None:
            self.run_button.state(["!disabled"])
        self.status.set(f"Language enrichment finished with exit code {return_code}.")
        self._append_log(f"Language enrichment finished with exit code {return_code}.\n")

    def _failed(self, message: str) -> None:
        if self.run_button is not None:
            self.run_button.state(["!disabled"])
        self.status.set("Language enrichment failed to start.")
        self._append_log(f"ERROR: {message}\n")
        messagebox.showerror("Language enrichment failed", message)

    def _append_log(self, value: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, value)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    TrajectoryLanguageStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
