"""Making models trained elsewhere load here.

Training happens on Colab and the console runs on a laptop, so a pickle
written by one scikit-learn has to open under another. Pickle records a
class by module path and name, and those paths are not stable across
versions: a Cython extension that reported itself as `_loss` in one release
reports `sklearn._loss._loss` in the next, and joblib then raises
ModuleNotFoundError for a class that is present and identical.

The alternative was pinning both machines to one scikit-learn, which is a
worse trade: it makes a Colab upgrade break the console silently, and it
does not help a model someone trained six months ago.

Nothing here converts or rewrites a model. It only tells pickle where a
class moved to, and only for names this project actually trains.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List

#: Module path as recorded in some pickle -> where the same class lives now.
#: Both directions are listed, because the console may be older or newer than
#: whatever trained the model.
_ALIASES: Dict[str, List[str]] = {
    # scikit-learn's loss extension. Bare `_loss` is what the build produced
    # for several releases; the qualified name is what it produces now.
    "_loss": ["sklearn._loss._loss"],
    "sklearn._loss._loss": ["_loss"],
}

_installed = False


def install_pickle_aliases() -> Dict[str, str]:
    """Point missing module paths at the modules that replaced them.

    Returns the aliases actually installed, which is empty on a machine whose
    scikit-learn matches whatever wrote the pickle. Safe to call repeatedly
    and safe to call when scikit-learn is absent.
    """
    global _installed
    done: Dict[str, str] = {}
    if _installed:
        return done

    for missing, candidates in _ALIASES.items():
        if missing in sys.modules:
            continue
        try:
            importlib.import_module(missing)
            continue                      # it is really there, leave it alone
        except Exception:
            pass
        for target in candidates:
            try:
                sys.modules[missing] = importlib.import_module(target)
                done[missing] = target
                break
            except Exception:
                continue

    _installed = True
    return done
