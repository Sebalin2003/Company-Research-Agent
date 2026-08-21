from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import PROJECT_ROOT, Settings, get_settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.domain.cv import extract_candidate_signals, validate_cv_text
from backend.app.domain.jobs import extract_job_signals, job_signal_payload
from backend.app.services.cv_file_extractor import CVExtractionError, extract_cv_text


class CVLibraryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CVLibraryService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        configured = Path(self.settings.cv_storage_dir)
        self.root = (configured if configured.is_absolute() else PROJECT_ROOT / configured).resolve()

    def list_cvs(self) -> list[models.StoredCV]:
        return list(
            self.db.scalars(
                select(models.StoredCV).order_by(
                    models.StoredCV.is_default.desc(), models.StoredCV.updated_at.desc()
                )
            )
        )

    def get_cv(self, cv_id: str) -> models.StoredCV | None:
        return self.db.get(models.StoredCV, cv_id)

    def get_version(self, cv_id: str, version_id: str) -> models.CVVersion | None:
        version = self.db.get(models.CVVersion, version_id)
        return version if version and version.cv_id == cv_id else None

    def versions(self, cv_id: str) -> list[models.CVVersion]:
        return list(
            self.db.scalars(
                select(models.CVVersion)
                .where(models.CVVersion.cv_id == cv_id)
                .order_by(models.CVVersion.version_number.desc())
            )
        )

    def current_version(self, cv: models.StoredCV) -> models.CVVersion:
        version = self.db.get(models.CVVersion, cv.current_version_id) if cv.current_version_id else None
        if not version or version.cv_id != cv.id:
            raise CVLibraryError("cv_version_not_found", "La versión actual del CV no está disponible.")
        return version

    def default_or_only(self) -> models.StoredCV | None:
        cvs = self.list_cvs()
        return next((item for item in cvs if item.is_default), None) or (cvs[0] if len(cvs) == 1 else None)

    def create_upload(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
        display_name: str | None = None,
        cv: models.StoredCV | None = None,
    ) -> tuple[models.StoredCV, models.CVVersion]:
        if len(content) > self.settings.cv_file_max_bytes:
            raise CVLibraryError("cv_file_too_large", "El archivo supera el límite de 10 MB.")
        safe_name = Path(filename or "cv").name
        try:
            extracted = extract_cv_text(
                safe_name,
                content_type,
                content,
                self.settings.cv_text_max_characters,
            )
        except CVExtractionError as exc:
            raise CVLibraryError("invalid_cv_file", str(exc)) from exc

        now = utc_now()
        if cv is None:
            requested_name = (display_name or "").strip()
            fallback_name = Path(safe_name).stem.strip() or "Mi CV"
            cv = models.StoredCV(
                id=str(uuid4()),
                display_name=(requested_name or fallback_name)[:200],
                is_default=(self.db.scalar(select(func.count(models.StoredCV.id))) or 0) == 0,
                current_version_id=None,
                created_at=now,
                updated_at=now,
            )
            self.db.add(cv)
            self.db.flush()
            created_from = "upload"
        else:
            created_from = "replacement"

        version_id = str(uuid4())
        suffix = Path(safe_name).suffix.lower()
        storage_key = f"{cv.id}/{version_id}{suffix}"
        target = self.resolve_storage_key(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

        number = int(
            self.db.scalar(
                select(func.max(models.CVVersion.version_number)).where(models.CVVersion.cv_id == cv.id)
            )
            or 0
        ) + 1
        version = models.CVVersion(
            id=version_id,
            cv_id=cv.id,
            version_number=number,
            storage_key=storage_key,
            original_filename=safe_name,
            content_type=content_type,
            file_size=len(content),
            file_hash=hashlib.sha256(content).hexdigest(),
            extracted_text=extracted.text,
            structured_profile_json=json.dumps(asdict(extract_candidate_signals(extracted.text)), ensure_ascii=False),
            created_from=created_from,
            source_version_id=cv.current_version_id,
            created_at=now,
        )
        self.db.add(version)
        cv.current_version_id = version.id
        cv.updated_at = now
        self.db.flush()
        return cv, version

    def create_text_version(
        self,
        cv: models.StoredCV,
        text: str,
        source_version_id: str | None = None,
    ) -> models.CVVersion:
        try:
            cleaned = validate_cv_text(text, self.settings.cv_text_max_characters)
        except ValueError as exc:
            raise CVLibraryError("invalid_cv_text", str(exc)) from exc
        if cleaned is None:
            raise CVLibraryError("invalid_cv_text", "El CV no puede estar vacío.")
        number = int(
            self.db.scalar(
                select(func.max(models.CVVersion.version_number)).where(models.CVVersion.cv_id == cv.id)
            )
            or 0
        ) + 1
        now = utc_now()
        version = models.CVVersion(
            id=str(uuid4()),
            cv_id=cv.id,
            version_number=number,
            storage_key=None,
            original_filename=None,
            content_type=None,
            file_size=0,
            file_hash=None,
            extracted_text=cleaned,
            structured_profile_json=json.dumps(asdict(extract_candidate_signals(cleaned)), ensure_ascii=False),
            created_from="text_edit",
            source_version_id=source_version_id or cv.current_version_id,
            created_at=now,
        )
        self.db.add(version)
        cv.current_version_id = version.id
        cv.updated_at = now
        self.db.flush()
        return version

    def set_default(self, cv: models.StoredCV) -> None:
        for item in self.list_cvs():
            item.is_default = item.id == cv.id
        cv.updated_at = utc_now()
        self.db.flush()

    def create_job_description(
        self,
        conversation: models.Conversation,
        message: models.ConversationMessage,
        title: str | None,
        text: str,
    ) -> models.JobDescription:
        cleaned = text.strip()
        if not cleaned:
            raise CVLibraryError("invalid_job_description", "La descripción del puesto no puede estar vacía.")
        if len(cleaned) > 20_000:
            raise CVLibraryError("invalid_job_description", "La descripción del puesto supera el límite permitido.")
        job = models.JobDescription(
            id=str(uuid4()),
            conversation_id=conversation.id,
            message_id=message.id,
            title=(title or "Descripción del puesto").strip()[:200],
            raw_text=cleaned,
            structured_signals_json=json.dumps(job_signal_payload(extract_job_signals(cleaned)), ensure_ascii=False),
            created_at=utc_now(),
        )
        self.db.add(job)
        self.db.flush()
        return job

    def review_recommendation(
        self,
        artifact: models.CVRecommendationArtifact,
        decisions: list[dict],
        draft_text: str | None,
        save_as_cv_version: bool,
    ) -> models.CVVersion | None:
        payload = json.loads(artifact.payload_json or "{}")
        suggestions = payload.get("change_suggestions") or []
        by_id = {str(item.get("id")): item for item in suggestions}
        decision_by_id = {str(item.get("suggestion_id")): item for item in decisions}
        if set(decision_by_id) != set(by_id):
            raise CVLibraryError(
                "invalid_cv_review",
                "La revisión debe aceptar, rechazar o editar cada sugerencia.",
            )
        for suggestion_id, decision in decision_by_id.items():
            choice = str(decision.get("decision") or "")
            if choice not in {"accepted", "rejected", "edited"}:
                raise CVLibraryError("invalid_cv_review", "Una decisión de revisión no es válida.")
            if choice == "edited" and not str(decision.get("edited_text") or "").strip():
                raise CVLibraryError("invalid_cv_review", "Una sugerencia editada requiere texto.")
            if (
                by_id[suggestion_id].get("type") == "add_only_if_true"
                and choice in {"accepted", "edited"}
                and decision.get("truth_confirmed") is not True
            ):
                raise CVLibraryError(
                    "cv_truth_confirmation_required",
                    "Las incorporaciones nuevas requieren confirmación explícita.",
                )

        cv = self.get_cv(artifact.cv_id)
        version = self.get_version(artifact.cv_id, artifact.cv_version_id)
        if not cv or not version:
            raise CVLibraryError("cv_not_found", "El CV usado para la revisión ya no existe.")
        applied = []
        manual = []
        generated_draft = version.extracted_text
        for suggestion_id, decision in decision_by_id.items():
            suggestion = by_id[suggestion_id]
            if decision["decision"] not in {"accepted", "edited"} or suggestion.get("type") != "rewrite":
                continue
            original = str(suggestion.get("original_text") or "")
            replacement = str(decision.get("edited_text") or suggestion.get("suggested_text") or "")
            if original and generated_draft.count(original) == 1 and is_direct_replacement(replacement):
                generated_draft = generated_draft.replace(original, replacement, 1)
                applied.append(suggestion_id)
            else:
                manual.append(suggestion_id)
        reviewed_draft = (draft_text if draft_text is not None else generated_draft).strip()
        if not reviewed_draft:
            raise CVLibraryError("invalid_cv_text", "El borrador revisado no puede estar vacío.")
        review = {
            "decisions": decisions,
            "automatically_applied_suggestion_ids": applied,
            "manual_suggestion_ids": manual,
            "saved_as_cv_version": save_as_cv_version,
        }
        artifact.review_json = json.dumps(review, ensure_ascii=False)
        artifact.draft_text = reviewed_draft
        artifact.status = "reviewed"
        artifact.updated_at = utc_now()
        saved = self.create_text_version(cv, reviewed_draft, version.id) if save_as_cv_version else None
        self.db.flush()
        return saved

    def resolve_storage_key(self, storage_key: str) -> Path:
        candidate = (self.root / storage_key).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise CVLibraryError("invalid_storage_key", "La ruta administrada del CV no es válida.") from exc
        return candidate

    def delete_version(self, cv: models.StoredCV, version: models.CVVersion) -> list[Path]:
        versions = self.versions(cv.id)
        if len(versions) == 1:
            raise CVLibraryError("last_cv_version", "No se puede eliminar la única versión del CV.")
        self._delete_recommendations(cv.id, version.id)
        self.db.delete(version)
        remaining = [item for item in versions if item.id != version.id]
        if cv.current_version_id == version.id:
            cv.current_version_id = remaining[0].id
            self._update_active_cv_context(cv.id, remaining[0].id)
        cv.updated_at = utc_now()
        self.db.flush()
        return [self.resolve_storage_key(version.storage_key)] if version.storage_key else []

    def delete_cv(self, cv: models.StoredCV) -> list[Path]:
        paths = [self.resolve_storage_key(item.storage_key) for item in self.versions(cv.id) if item.storage_key]
        self._delete_recommendations(cv.id)
        links = list(
            self.db.scalars(
                select(models.ConversationArtifact).where(
                    models.ConversationArtifact.artifact_type == "cv",
                    models.ConversationArtifact.artifact_id == cv.id,
                )
            )
        )
        for link in links:
            self.db.delete(link)
        self._update_active_cv_context(cv.id, None)
        self.db.delete(cv)
        self.db.flush()
        remaining = self.list_cvs()
        if len(remaining) == 1:
            remaining[0].is_default = True
        self.db.flush()
        return paths

    def _delete_recommendations(self, cv_id: str, version_id: str | None = None) -> None:
        statement = select(models.CVRecommendationArtifact).where(
            models.CVRecommendationArtifact.cv_id == cv_id
        )
        if version_id:
            statement = statement.where(models.CVRecommendationArtifact.cv_version_id == version_id)
        recommendations = list(self.db.scalars(statement))
        ids = [item.id for item in recommendations]
        if ids:
            links = list(
                self.db.scalars(
                    select(models.ConversationArtifact).where(
                        models.ConversationArtifact.artifact_type == "cv_recommendation",
                        models.ConversationArtifact.artifact_id.in_(ids),
                    )
                )
            )
            for link in links:
                self.db.delete(link)
        for recommendation in recommendations:
            self.db.delete(recommendation)

    def _update_active_cv_context(self, cv_id: str, version_id: str | None) -> None:
        conversations = list(self.db.scalars(select(models.Conversation)))
        for conversation in conversations:
            context = json.loads(conversation.active_context_json or "{}")
            if context.get("active_cv_id") != cv_id:
                continue
            if version_id:
                context["active_cv_version_id"] = version_id
            else:
                context.pop("active_cv_id", None)
                context.pop("active_cv_version_id", None)
            conversation.active_context_json = json.dumps(context, ensure_ascii=False)


def profile_payload(version: models.CVVersion) -> dict:
    return json.loads(version.structured_profile_json or "{}")


def is_direct_replacement(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized) and not normalized.startswith(
        ("reescribir", "redactar", "cambiar", "adaptar", "modificar", "destacar", "enfatizar", "agregar")
    )


def remove_managed_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
