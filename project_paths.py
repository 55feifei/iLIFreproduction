"""Portable paths shared by training and the dependency-free queue planner."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def esc_data_path(value=None):
    explicit = value or os.environ.get('ILIF_ESC_DATA')
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [ROOT / 'data/esc', ROOT / 'data', ROOT / 'reference_source/data/database',
                  PROJECT / 'reference_source/data/database']
    filename = 'EventStoryLine_bert_intra_and_inter_train_features.pkl'
    return next((path for path in candidates if (path / filename).is_file()), candidates[0])


ESC_DATA = esc_data_path()
# The original, optional MAVEN preprocessing sources must not be required just
# to import common.py on a different machine.
AUTHOR = Path(os.environ['ILIF_AUTHOR_SOURCE']).expanduser().resolve() if os.environ.get('ILIF_AUTHOR_SOURCE') else next(
    PROJECT.glob('刘程工作/2.实验代码/2.1*'), PROJECT / 'author_source')
