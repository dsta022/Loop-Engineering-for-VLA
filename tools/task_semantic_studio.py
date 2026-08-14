#!/usr/bin/env python3
"""Local window for parsing, reviewing, and using task semantic profiles."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from utils.task_description_parser import load_parser, load_parser_config, normalize_parser_result
from utils.task_semantic_profile import TaskSemanticProfile


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PARSER = "utils.task_description_parser:HeuristicTaskDescriptionParser"


def build_audit_command(
    dataset: Path,
    profile_dir: Path | None,
    evaluator_spec: str,
    out_dir: Path | None = None,
    evaluator_config: Path | None = None,
    review_threshold: float = 0.5,
    python_executable: str = sys.executable,
    pass_threshold: float = 0.9,
) -> list[str]:
    if not evaluator_spec.strip():
        raise ValueError("semantic evaluator module:attribute is required")
    if not 0.0 <= review_threshold <= 1.0:
        raise ValueError("review threshold must be within [0, 1]")
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError("pass threshold must be within [0, 1]")
    if review_threshold > pass_threshold:
        raise ValueError("review threshold must not exceed pass threshold")
    command = [
        python_executable,
        str(REPO_ROOT / "data_quality_audit.py"),
        str(dataset),
        "--semantic-evaluator",
        evaluator_spec.strip(),
        "--semantic-failure-threshold",
        str(review_threshold),
        "--semantic-pass-threshold",
        str(pass_threshold),
    ]
    if profile_dir is not None:
        command.extend(["--task-profile-dir", str(profile_dir)])
    if out_dir is not None:
        command.extend(["--out-dir", str(out_dir)])
    if evaluator_config is not None:
        command.extend(["--semantic-config", str(evaluator_config)])
    return command


class TaskSemanticStudio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Task Semantic Studio")
        self.root.geometry("1280x850")
        self.root.minsize(980, 700)

        self.task_name = tk.StringVar()
        self.task_key = tk.StringVar()
        self.parser_spec = tk.StringVar(value=DEFAULT_PARSER)
        self.parser_config = tk.StringVar()
        self.profile_dir = tk.StringVar()
        self.dataset = tk.StringVar()
        self.report_dir = tk.StringVar()
        self.evaluator_spec = tk.StringVar()
        self.evaluator_config = tk.StringVar()
        self.review_threshold = tk.StringVar(value="0.5")
        self.pass_threshold = tk.StringVar(value="0.9")
        self.status = tk.StringVar(value="Ready")
        self.run_button: ttk.Button | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(2, weight=4)
        root_frame.rowconfigure(5, weight=2)

        identity = ttk.LabelFrame(root_frame, text="Task identity", padding=8)
        identity.grid(row=0, column=0, sticky="ew")
        identity.columnconfigure(1, weight=1)
        identity.columnconfigure(3, weight=1)
        ttk.Label(identity, text="Dataset task text / name").grid(row=0, column=0, sticky="w")
        ttk.Entry(identity, textvariable=self.task_name).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(identity, text="Task key (optional)").grid(row=0, column=2, sticky="w")
        ttk.Entry(identity, textvariable=self.task_key).grid(row=0, column=3, sticky="ew", padx=6)

        parser = ttk.LabelFrame(root_frame, text="Description parser", padding=8)
        parser.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        parser.columnconfigure(1, weight=1)
        parser.columnconfigure(3, weight=1)
        ttk.Label(parser, text="Parser module:attribute").grid(row=0, column=0, sticky="w")
        ttk.Entry(parser, textvariable=self.parser_spec).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(parser, text="Parser config JSON").grid(row=0, column=2, sticky="w")
        ttk.Entry(parser, textvariable=self.parser_config).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(parser, text="Browse", command=self._browse_parser_config).grid(row=0, column=4)

        panes = ttk.Panedwindow(root_frame, orient=tk.HORIZONTAL)
        panes.grid(row=2, column=0, sticky="nsew", pady=8)

        description_frame = ttk.LabelFrame(panes, text="Detailed task description", padding=6)
        profile_frame = ttk.LabelFrame(panes, text="Parsed profile JSON (editable)", padding=6)
        panes.add(description_frame, weight=1)
        panes.add(profile_frame, weight=1)
        self.description_text = tk.Text(description_frame, wrap=tk.WORD, undo=True)
        self.description_text.pack(fill=tk.BOTH, expand=True)
        self.profile_text = tk.Text(profile_frame, wrap=tk.NONE, undo=True)
        self.profile_text.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(root_frame)
        actions.grid(row=3, column=0, sticky="ew")
        ttk.Button(actions, text="Parse description", command=self.parse_description).pack(side=tk.LEFT)
        ttk.Button(actions, text="Validate profile", command=self.validate_profile).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Load profile", command=self.load_profile).pack(side=tk.LEFT)
        ttk.Button(actions, text="Save profile", command=self.save_profile).pack(side=tk.LEFT, padx=6)

        audit = ttk.LabelFrame(root_frame, text="Run semantic quality audit", padding=8)
        audit.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        audit.columnconfigure(1, weight=1)
        audit.columnconfigure(4, weight=1)
        self._path_row(audit, 0, "Dataset", self.dataset, self._browse_dataset)
        self._path_row(
            audit,
            1,
            "Task Profile directory (optional)",
            self.profile_dir,
            self._browse_profile_dir,
        )
        self._path_row(audit, 2, "Report directory", self.report_dir, self._browse_report_dir)
        ttk.Label(audit, text="Evaluator module:attribute").grid(row=3, column=0, sticky="w")
        ttk.Entry(audit, textvariable=self.evaluator_spec).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=2
        )
        ttk.Label(audit, text="Evaluator config JSON").grid(row=3, column=3, sticky="w")
        ttk.Entry(audit, textvariable=self.evaluator_config).grid(row=3, column=4, sticky="ew", padx=6)
        ttk.Button(audit, text="Browse", command=self._browse_evaluator_config).grid(row=3, column=5)
        ttk.Label(audit, text="Likely failure threshold").grid(row=4, column=0, sticky="w")
        ttk.Entry(audit, textvariable=self.review_threshold, width=12).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(audit, text="Automatic pass threshold").grid(row=4, column=3, sticky="w")
        ttk.Entry(audit, textvariable=self.pass_threshold, width=12).grid(
            row=4, column=4, sticky="w", padx=6
        )
        self.run_button = ttk.Button(audit, text="Run audit", command=self.run_audit)
        self.run_button.grid(row=4, column=5, sticky="e")

        log_frame = ttk.LabelFrame(root_frame, text="Output", padding=6)
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(root_frame, textvariable=self.status).grid(row=6, column=0, sticky="w", pady=(4, 0))

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=4, sticky="ew", padx=6, pady=2
        )
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=5)

    def _browse_parser_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.parser_config.set(path)

    def _browse_evaluator_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.evaluator_config.set(path)

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.dataset.set(path)

    def _browse_profile_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.profile_dir.set(path)

    def _browse_report_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.report_dir.set(path)

    def _profile_from_editor(self) -> TaskSemanticProfile:
        payload = json.loads(self.profile_text.get("1.0", tk.END))
        if not isinstance(payload, dict):
            raise TypeError("profile editor must contain one JSON object")
        return TaskSemanticProfile.from_dict(payload)

    def _show_profile(self, profile: TaskSemanticProfile) -> None:
        self.profile_text.delete("1.0", tk.END)
        self.profile_text.insert("1.0", json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))

    def parse_description(self) -> None:
        try:
            config_path = Path(self.parser_config.get()) if self.parser_config.get().strip() else None
            parser = load_parser(self.parser_spec.get().strip(), load_parser_config(config_path))
            profile = normalize_parser_result(
                parser.parse(
                    self.description_text.get("1.0", tk.END),
                    task_name=self.task_name.get(),
                    task_key=self.task_key.get(),
                )
            )
            self._show_profile(profile)
            self.task_name.set(profile.task_name)
            self.task_key.set(profile.task_key)
            self.status.set("Description parsed. Review and validate the JSON before saving.")
        except Exception as exc:
            messagebox.showerror("Parse failed", str(exc))

    def validate_profile(self) -> None:
        try:
            profile = self._profile_from_editor()
            profile.validate()
            self.status.set(f"Profile {profile.task_key!r} is valid.")
            messagebox.showinfo("Valid profile", f"{profile.task_key} passed validation.")
        except Exception as exc:
            messagebox.showerror("Invalid profile", str(exc))

    def load_profile(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Task profile", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            profile = TaskSemanticProfile.load(Path(path))
            self._show_profile(profile)
            self.task_name.set(profile.task_name)
            self.task_key.set(profile.task_key)
            self.profile_dir.set(str(Path(path).parent))
            self.status.set(f"Loaded {path}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def save_profile(self) -> None:
        try:
            profile = self._profile_from_editor()
            directory_value = self.profile_dir.get().strip()
            if not directory_value:
                directory_value = filedialog.askdirectory(title="Select directory for task profiles")
                if not directory_value:
                    return
                self.profile_dir.set(directory_value)
            directory = Path(directory_value).expanduser()
            path = directory / f"{profile.task_key}.json"
            profile.save(path)
            self.status.set(f"Saved {path}")
            self._append_log(f"Saved validated task profile: {path}\n")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def run_audit(self) -> None:
        try:
            dataset = Path(self.dataset.get()).expanduser()
            profile_dir = (
                Path(self.profile_dir.get()).expanduser() if self.profile_dir.get().strip() else None
            )
            report_dir = Path(self.report_dir.get()).expanduser() if self.report_dir.get().strip() else None
            evaluator_config = (
                Path(self.evaluator_config.get()).expanduser()
                if self.evaluator_config.get().strip()
                else None
            )
            command = build_audit_command(
                dataset=dataset,
                profile_dir=profile_dir,
                evaluator_spec=self.evaluator_spec.get(),
                out_dir=report_dir,
                evaluator_config=evaluator_config,
                review_threshold=float(self.review_threshold.get()),
                pass_threshold=float(self.pass_threshold.get()),
            )
            if not dataset.is_dir():
                raise FileNotFoundError(f"dataset directory not found: {dataset}")
            if profile_dir is not None and not profile_dir.is_dir():
                raise FileNotFoundError(f"profile directory not found: {profile_dir}")
        except Exception as exc:
            messagebox.showerror("Cannot run audit", str(exc))
            return

        if self.run_button is not None:
            self.run_button.state(["disabled"])
        self.status.set("Audit running...")
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
            self.root.after(0, self._audit_finished, return_code)
        except Exception as exc:
            self.root.after(0, self._audit_failed, str(exc))

    def _audit_finished(self, return_code: int) -> None:
        if self.run_button is not None:
            self.run_button.state(["!disabled"])
        self.status.set(f"Audit finished with exit code {return_code}.")
        self._append_log(f"Audit finished with exit code {return_code}.\n")

    def _audit_failed(self, message: str) -> None:
        if self.run_button is not None:
            self.run_button.state(["!disabled"])
        self.status.set("Audit failed to start.")
        self._append_log(f"ERROR: {message}\n")
        messagebox.showerror("Audit failed", message)

    def _append_log(self, value: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, value)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    TaskSemanticStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
