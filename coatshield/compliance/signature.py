"""Operator e-signature: confirming a stop or a setpoint needs a name, the PIN re-entered,
a reason code and the meaning of the signature. The system only recommends; it never
writes a setpoint by itself, and it has no connection to any safety interlock.
"""

from __future__ import annotations

import hashlib
import hmac

from coatshield.compliance.audit import AuditTrail
from coatshield.config import Config, UserCfg


class SignatureError(Exception):
    """The signature was refused (unknown user, wrong PIN, missing reason or meaning)."""


def hash_pin(name: str, pin: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), f"coatshield:{name}".encode(),
                               iterations).hex()


def _find_user(cfg: Config, name: str) -> UserCfg | None:
    return next((u for u in cfg.compliance.users if u.name == name), None)


def sign(trail: AuditTrail, cfg: Config, user: str, pin: str, meaning: str, reason_code: str,
         obj: str, old=None, new=None, comment: str = "", model_hash: str = "",
         config_hash: str = "") -> dict:
    """Record a signed operator decision in the audit trail and return the entry.

    A refused attempt is logged too, without the PIN.
    """
    cc = cfg.compliance
    account = _find_user(cfg, user)
    problem = None
    if meaning not in cc.meanings:
        problem = f"unknown meaning '{meaning}'"
    elif reason_code not in cc.reason_codes:
        problem = f"unknown reason code '{reason_code}'"
    elif account is None:
        problem = "unknown user"
    elif not hmac.compare_digest(hash_pin(user, pin, cc.pin_iterations), account.pin_hash):
        problem = "PIN does not match"
    if problem:
        trail.append(user, account.role if account else "unknown", "signature_refused", obj,
                     reason=problem, model_hash=model_hash, config_hash=config_hash)
        raise SignatureError(problem)
    reason = reason_code if not comment else f"{reason_code}: {comment}"
    return trail.append(
        user, account.role, meaning, obj, old=old, new=new, reason=reason,
        model_hash=model_hash, config_hash=config_hash,
        signature={"signed_by": user, "role": account.role, "meaning": meaning,
                   "reason_code": reason_code},
    )
