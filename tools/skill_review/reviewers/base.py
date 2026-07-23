"""The read-only reviewer interface and the review subject.

A ``Reviewer`` is a pure judge (INV-4): it reads a ``SkillWrite`` and returns a
``Verdict`` with no side effects — it must not touch the filesystem, config,
memory, or the live structural validators. It expresses a rubric's intent as a
verdict; it never mutates or rolls back anything (INV-6).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tools.skill_review.schema import GraderType, Verdict


@dataclass(frozen=True)
class SkillWrite:
    """A proposed agent skill write under review (mirrors ``skill_manage`` kwargs).

    This is the reviewers' *input* type; it is internal and may grow additively.
    It is NOT part of the frozen persisted schema (that is ``schema.py``).
    """

    action: str
    name: str
    content: str | None = None          # full SKILL.md text (create / edit)
    file_path: str | None = None        # write_file / remove_file / patch-on-file
    file_content: str | None = None      # write_file
    old_string: str | None = None        # patch
    new_string: str | None = None        # patch
    origin: str = "background_review"


class Reviewer(ABC):
    """Base class for orthogonal reviewers.

    Subclasses set two class attributes:
      * ``id`` — a stable reviewer identifier (e.g. ``"contract"``).
      * ``grader_type`` — how it grades; drives panel ordering (deterministic-first).
    """

    id: str = ""
    grader_type: GraderType = GraderType.DETERMINISTIC

    @abstractmethod
    def review(self, write: SkillWrite) -> Verdict:
        """Return a ``Verdict`` for ``write``. MUST be pure / side-effect free."""
        raise NotImplementedError
