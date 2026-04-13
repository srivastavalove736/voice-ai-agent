import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agent.schemas import IntentResult, ToolExecutionResult


class LocalToolExecutor:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.notes_path = workspace_root / "notes" / "local_notes.txt"
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(
        self, intents: List[IntentResult], transcript: str
    ) -> List[ToolExecutionResult]:
        results: List[ToolExecutionResult] = []
        for intent_res in intents:
            handler = self._get_handler(intent_res.intent)
            try:
                output = handler(transcript)
                results.append(
                    ToolExecutionResult(
                        tool_name=intent_res.intent, ok=True, output=output
                    )
                )
            except Exception as exc:  # pragma: no cover
                results.append(
                    ToolExecutionResult(
                        tool_name=intent_res.intent, ok=False, output=str(exc)
                    )
                )
        return results

    def _get_handler(self, intent: str):
        handlers = {
            "get_time": self._get_time,
            "list_files": self._list_files,
            "create_note": self._create_note,
            "system_info": self._system_info,
            "delete_notes": self._delete_notes,
            "help": self._help,
        }
        return handlers.get(intent, self._help)

    def _get_time(self, transcript: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current local time: {now}"

    def _list_files(self, transcript: str) -> str:
        entries = sorted(p.name for p in self.workspace_root.iterdir())
        if not entries:
            return "Workspace is empty"
        return "Workspace files/folders:\n- " + "\n- ".join(entries)

    def _create_note(self, transcript: str) -> str:
        note_text = self._extract_note_text(transcript)
        if not note_text:
            note_text = transcript

        confirm = input(f"Save note with text: '{note_text.strip()}'? [y/N] ")
        if confirm.lower().strip() != "y":
            return "Note creation cancelled."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {note_text.strip()}\n"
        with self.notes_path.open("a", encoding="utf-8") as f:
            f.write(line)

        return f"Saved note to {self.notes_path.name}: {note_text.strip()}"

    def _delete_notes(self, transcript: str) -> str:
        if not self.notes_path.exists():
            return "No notes file to delete."

        confirm = input(f"Delete all notes at '{self.notes_path}'? [y/N] ")
        if confirm.lower().strip() != "y":
            return "Deletion cancelled."

        self.notes_path.unlink()
        return "All notes have been deleted."

    def _system_info(self, transcript: str) -> str:
        info: Dict[str, str] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        }
        lines = [f"{key}: {value}" for key, value in info.items()]
        return "System information:\n" + "\n".join(lines)

    def _help(self, transcript: str) -> str:
        return (
            "Supported intents: get_time, list_files, create_note, delete_notes, system_info, help. "
            "Try saying commands like 'what time is it' or 'create a note saying finish report'."
        )

    @staticmethod
    def _extract_note_text(transcript: str) -> str:
        text = transcript.strip()
        candidates = [
            "note saying",
            "note that",
            "remember this",
            "write this down",
            "save note",
            "save a note",
            "take a note",
        ]

        lowered = text.lower()
        for marker in candidates:
            idx = lowered.find(marker)
            if idx >= 0:
                return text[idx + len(marker) :].strip(" :,-")

        return text

