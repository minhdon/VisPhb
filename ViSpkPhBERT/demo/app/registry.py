"""Model registry and path discovery for the ViSPhB Streamlit demo.

The registry intentionally keeps path logic outside the UI.  It supports the
current repository layout under ``models/`` and a legacy/local layout under
``uit-models/`` without hard-coding an absolute machine path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SEEDS: Tuple[int, ...] = (42, 52, 62)
SPIKING_LAYER_OPTIONS: Tuple[int, ...] = (2, 4)
DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR.parent
PROJECT_ROOT = DEMO_DIR.parent
CODE_DIR = PROJECT_ROOT / "code"


@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    short_label: str
    description: str
    architecture: str
    module_path: Optional[Path]
    class_name: Optional[str]
    config_class: Optional[str]
    uses_stage_config: bool
    supports_seed: bool
    spiking_options: Tuple[int, ...]
    checkpoint_prefix: str
    report_prefix: str


@dataclass(frozen=True)
class ModelVariant:
    family_key: str
    family_label: str
    short_label: str
    description: str
    architecture: str
    model_root: Path
    run_dir: Path
    checkpoint_path: Optional[Path]
    report_paths: Tuple[Path, ...]
    seed: Optional[int]
    spiking_layers: Optional[int]
    module_path: Optional[Path]
    class_name: Optional[str]
    config_class: Optional[str]
    uses_stage_config: bool
    model_name: str = DEFAULT_MODEL_NAME

    @property
    def variant_id(self) -> str:
        seed = "na" if self.seed is None else str(self.seed)
        spk = "na" if self.spiking_layers is None else str(self.spiking_layers)
        return f"{self.family_key}|seed={seed}|spk={spk}|root={self.model_root}"

    @property
    def display_name(self) -> str:
        parts = [self.family_label]
        if self.seed is not None:
            parts.append(f"seed {self.seed}")
        if self.spiking_layers is not None:
            parts.append(f"spk{self.spiking_layers}")
        return " · ".join(parts)

    @property
    def compact_name(self) -> str:
        if self.seed is None and self.spiking_layers is None:
            return self.short_label
        suffixes: List[str] = []
        if self.seed is not None:
            suffixes.append(f"s{self.seed}")
        if self.spiking_layers is not None:
            suffixes.append(f"spk{self.spiking_layers}")
        return f"{self.short_label} ({', '.join(suffixes)})"

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_path is not None and self.checkpoint_path.exists()

    @property
    def has_reports(self) -> bool:
        return any(path.exists() for path in self.report_paths)


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "teacher": FamilySpec(
        key="teacher",
        label="Teacher",
        short_label="PhoBERT teacher",
        description="PhoBERT baseline fine-tuned on UIT-VSFC.",
        architecture="teacher",
        module_path=None,
        class_name=None,
        config_class=None,
        uses_stage_config=False,
        supports_seed=True,
        spiking_options=(),
        checkpoint_prefix="best_model",
        report_prefix="phobert_vsfc",
    ),
    "tier1_direct": FamilySpec(
        key="tier1_direct",
        label="Tier 1 Direct",
        short_label="Tier1 Direct",
        description="SpikeBERT-style student trained directly on UIT-VSFC.",
        architecture="tier1_direct",
        module_path=CODE_DIR / "tier1_spikebert.py",
        class_name="SpikeBERTStudent",
        config_class="Config",
        uses_stage_config=False,
        supports_seed=True,
        spiking_options=(4,),
        checkpoint_prefix="best_tier1",
        report_prefix="energy_report",
    ),
    "tier1_twostage": FamilySpec(
        key="tier1_twostage",
        label="Tier 1 Two-stage",
        short_label="Tier1 2-stage",
        description="Stage 1 Wiki KD followed by task KD with BPTT/surrogate gradients.",
        architecture="tier1_twostage",
        module_path=CODE_DIR / "tier1_spikebert_2stage.py",
        class_name="SpikeBERTStudent",
        config_class="Config",
        uses_stage_config=True,
        supports_seed=True,
        spiking_options=SPIKING_LAYER_OPTIONS,
        checkpoint_prefix="best_tier1_spikebert_2stage",
        report_prefix="tier1_spikebert_2stage_report",
    ),
    "tier2_twostage": FamilySpec(
        key="tier2_twostage",
        label="Tier 2 Two-stage",
        short_label="Tier2 2-stage",
        description="Stage 1 Wiki KD followed by implicit/fixed-point task KD.",
        architecture="tier2_twostage",
        module_path=CODE_DIR / "tier2_implicit_2stage.py",
        class_name="ImplicitPhoBERTStudent",
        config_class="Config",
        uses_stage_config=True,
        supports_seed=True,
        spiking_options=SPIKING_LAYER_OPTIONS,
        checkpoint_prefix="best_tier2_implicit_2stage",
        report_prefix="tier2_implicit_2stage_report",
    ),
    "tier3_twostage": FamilySpec(
        key="tier3_twostage",
        label="Tier 3 Two-stage",
        short_label="Tier3 2-stage",
        description="Stage 1 Wiki KD followed by hybrid implicit task KD.",
        architecture="tier3_twostage",
        module_path=CODE_DIR / "tier3_hybrid_2stage.py",
        class_name="HybridImplicitPhoBERTStudent",
        config_class="Config",
        uses_stage_config=True,
        supports_seed=True,
        spiking_options=SPIKING_LAYER_OPTIONS,
        checkpoint_prefix="best_tier3_hybrid_2stage",
        report_prefix="tier3_hybrid_2stage_report",
    ),
}


def family_options() -> Dict[str, str]:
    return {key: spec.label for key, spec in FAMILY_SPECS.items()}


def default_model_root() -> str:
    return os.getenv("VISPHB_MODEL_ROOT", "models")


def resolve_model_root(value: str | Path) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _existing_first(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _glob_many(base: Path, patterns: Iterable[str]) -> List[Path]:
    if not base.exists():
        return []
    results: List[Path] = []
    for pattern in patterns:
        results.extend(base.glob(pattern))
    return sorted({path for path in results if path.is_file()})


def _prefer_checkpoint(files: Sequence[Path]) -> Optional[Path]:
    if not files:
        return None

    def score(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        if name.startswith("best_") or "best" in name:
            return (0, name)
        if "last" in name:
            return (2, name)
        return (1, name)

    return sorted(files, key=score)[0]


def _teacher_run_dirs(root: Path) -> List[Path]:
    return [
        root / "baseline" / "phobert_vsfc",
        root / "phobert_vsfc",
        root / "baseline",
        root,
    ]


def _direct_run_dirs(root: Path, seed: int) -> List[Path]:
    return [
        root / "direct" / "tier1_spikebert" / f"seed_{seed}",
        root / "tier1_spikebert" / f"seed_{seed}",
        root / "direct" / "tier1_spikebert",
        root / "tier1_spikebert",
    ]


def _twostage_run_dirs(root: Path, family_key: str, seed: int, spiking_layers: int) -> List[Path]:
    folder = {
        "tier1_twostage": "tier1_spikebert_2stage",
        "tier2_twostage": "tier2_implicit_2stage",
        "tier3_twostage": "tier3_hybrid_2stage",
    }[family_key]

    spk_plain = f"spk{spiking_layers}"
    spk_t16 = f"spk{spiking_layers}_T16"
    return [
        root / "2-stage" / folder / spk_t16 / f"seed_{seed}",
        root / "2-stage" / folder / spk_plain / f"seed_{seed}",
        root / folder / spk_t16 / f"seed_{seed}",
        root / folder / spk_plain / f"seed_{seed}",
        root / folder / f"seed_{seed}",
        root / "2-stage" / folder / f"seed_{seed}",
    ]


def candidate_run_dirs(
    model_root: Path,
    family_key: str,
    seed: Optional[int],
    spiking_layers: Optional[int],
) -> List[Path]:
    if family_key == "teacher":
        return _teacher_run_dirs(model_root)
    if family_key == "tier1_direct":
        return _direct_run_dirs(model_root, int(seed or SEEDS[0]))
    if family_key in {"tier1_twostage", "tier2_twostage", "tier3_twostage"}:
        return _twostage_run_dirs(
            model_root,
            family_key,
            int(seed or SEEDS[0]),
            int(spiking_layers or SPIKING_LAYER_OPTIONS[0]),
        )
    raise KeyError(f"Unknown family: {family_key}")


def find_checkpoint(run_dir: Path, spec: FamilySpec, seed: Optional[int]) -> Optional[Path]:
    if spec.key == "teacher":
        return _prefer_checkpoint(
            _glob_many(run_dir, ["best_model.pth", "best_model.pt", "*.pth", "*.pt"])
        )

    seed_suffix = "" if seed is None else f"_seed{seed}"
    return _prefer_checkpoint(
        _glob_many(
            run_dir,
            [
                f"{spec.checkpoint_prefix}{seed_suffix}.pt",
                f"{spec.checkpoint_prefix}{seed_suffix}.pth",
                "best*.pt",
                "best*.pth",
                "*.pt",
                "*.pth",
            ],
        )
    )


def find_report_paths(run_dir: Path, spec: FamilySpec, seed: Optional[int]) -> Tuple[Path, ...]:
    if not run_dir.exists():
        return ()
    if spec.key == "teacher":
        preferred = [
            run_dir / "best_config.json",
            run_dir / "test_results.json",
            run_dir / "efficiency_report.json",
        ]
        others = _glob_many(run_dir, ["*.json"])
        return tuple(path for path in preferred if path.exists()) + tuple(
            path for path in others if path not in preferred
        )

    seed_suffix = "" if seed is None else f"_seed{seed}"
    preferred = _glob_many(
        run_dir,
        [
            f"{spec.report_prefix}{seed_suffix}.json",
            f"{spec.report_prefix}_seed{seed}.json" if seed is not None else "*.json",
            f"energy_report_seed{seed}.json" if seed is not None else "energy_report*.json",
            "*.json",
        ],
    )
    return tuple(preferred)


def get_variant(
    family_key: str,
    model_root_value: str | Path,
    seed: Optional[int] = None,
    spiking_layers: Optional[int] = None,
) -> ModelVariant:
    spec = FAMILY_SPECS[family_key]
    model_root = resolve_model_root(model_root_value)

    resolved_seed = seed if spec.supports_seed else None
    resolved_spk = spiking_layers if spec.spiking_options else None
    if spec.spiking_options and resolved_spk is None:
        resolved_spk = spec.spiking_options[0]

    run_dir = _existing_first(candidate_run_dirs(model_root, family_key, resolved_seed, resolved_spk))
    checkpoint = find_checkpoint(run_dir, spec, resolved_seed)
    reports = find_report_paths(run_dir, spec, resolved_seed)

    return ModelVariant(
        family_key=spec.key,
        family_label=spec.label,
        short_label=spec.short_label,
        description=spec.description,
        architecture=spec.architecture,
        model_root=model_root,
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        report_paths=reports,
        seed=resolved_seed,
        spiking_layers=resolved_spk,
        module_path=spec.module_path,
        class_name=spec.class_name,
        config_class=spec.config_class,
        uses_stage_config=spec.uses_stage_config,
    )


def iter_variants(model_root_value: str | Path, only_existing: bool = False) -> List[ModelVariant]:
    variants: List[ModelVariant] = []
    for family_key, spec in FAMILY_SPECS.items():
        seeds = SEEDS if spec.supports_seed else (None,)
        spiking_values: Sequence[Optional[int]] = spec.spiking_options or (None,)
        for seed in seeds:
            for spiking_layers in spiking_values:
                variant = get_variant(family_key, model_root_value, seed, spiking_layers)
                if only_existing and not (variant.has_checkpoint or variant.has_reports):
                    continue
                variants.append(variant)
    return variants


def variant_from_id(variant_id: str, variants: Sequence[ModelVariant]) -> ModelVariant:
    for variant in variants:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(f"Unknown variant id: {variant_id}")

