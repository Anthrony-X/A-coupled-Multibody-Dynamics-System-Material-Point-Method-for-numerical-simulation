from __future__ import annotations

from dataclasses import dataclass


CONSTITUTIVE_MODEL_DRUCKER_PRAGER = 0
CONSTITUTIVE_MODEL_MOHR_COULOMB = 1
CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY = 2
CONSTITUTIVE_MODEL_PURE_WATER = 3
CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP = 4


@dataclass(frozen=True)
class ConstitutiveModelInfo:
    name: str
    model_id: int
    aliases: tuple[str, ...]
    implemented: bool = True


_MODEL_INFOS = (
    ConstitutiveModelInfo(
        name="drucker-prager",
        model_id=CONSTITUTIVE_MODEL_DRUCKER_PRAGER,
        aliases=("dp", "drucker_prager", "drucker-prager"),
    ),
    ConstitutiveModelInfo(
        name="mohr-coulomb",
        model_id=CONSTITUTIVE_MODEL_MOHR_COULOMB,
        aliases=("mc", "mohr_coulomb", "mohr-coulomb"),
    ),
    ConstitutiveModelInfo(
        name="modified-cam-clay",
        model_id=CONSTITUTIVE_MODEL_MODIFIED_CAM_CLAY,
        aliases=("mcc", "modified_cam_clay", "modified-cam-clay", "cam-clay"),
    ),
    ConstitutiveModelInfo(
        name="pure-water",
        model_id=CONSTITUTIVE_MODEL_PURE_WATER,
        aliases=("water", "pure_water", "pure-water", "newtonian-water"),
    ),
    ConstitutiveModelInfo(
        name="srsh-modified-dp",
        model_id=CONSTITUTIVE_MODEL_SRSH_MODIFIED_DP,
        aliases=(
            "srsh",
            "srsh_modified_dp",
            "srsh-modified-dp",
            "improved-dp",
        ),
    ),
)

CONSTITUTIVE_MODEL_IDS = {info.name: info.model_id for info in _MODEL_INFOS}
_ALIASES = {
    alias.lower().replace("_", "-"): info.name
    for info in _MODEL_INFOS
    for alias in (info.name, *info.aliases)
}
CONSTITUTIVE_MODEL_CHOICES = tuple(sorted(_ALIASES))


def normalize_constitutive_model_name(name: str | None) -> str:
    key = "drucker-prager" if name is None else str(name).strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    available = ", ".join(CONSTITUTIVE_MODEL_CHOICES)
    raise ValueError(f"Unsupported soil constitutive model {name!r}. Available models: {available}")
