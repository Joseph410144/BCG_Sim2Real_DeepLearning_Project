"""Leakage-safe subject-level cross-validation splits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectFold:
    fold: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]

    def to_dict(self):
        return {
            "fold": self.fold,
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


def leave_one_subject_out_folds(subject_ids):
    """Create nested LOSO folds: one test subject and a different validation subject."""
    subjects = tuple(sorted(set(int(subject) for subject in subject_ids)))
    if len(subjects) < 3:
        raise ValueError("At least three distinct subjects are required")
    folds = []
    for index, test_subject in enumerate(subjects):
        val_subject = subjects[(index + 1) % len(subjects)]
        train_subjects = tuple(subject for subject in subjects if subject not in (test_subject, val_subject))
        folds.append(SubjectFold(index, train_subjects, (val_subject,), (test_subject,)))
    return folds
