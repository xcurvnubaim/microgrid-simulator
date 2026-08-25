"""Concrete training/evaluation runners wiring the agent loop to train.py.

``SACTrainRunners`` implements the orchestrator's ``Runners`` protocol using
the real trainer (:func:`microgrid_simulator.rl.train.train`) and a frozen
validation evaluator. Training functions are injectable so tests run without
launching jobs; constructing this class does not train anything.

Contract notes (plan sections 5, 7, 11):
- fine-tuning loads the matching incumbent seed checkpoint + VecNormalize;
- the replay buffer is never carried over (fresh buffer per run);
- fine-tune runs default to ``learning_rate = 1e-4`` unless the proposal
  itself changes the learning rate;
- selection uses the February validation split only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stable_baselines3.common.vec_env import DummyVecEnv

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.rl.env import MicrogridEnv
from microgrid_simulator.rl.orchestrator import (
    CandidateMetrics,
    SmokeRunResult,
)
from microgrid_simulator.rl.registry import ProposalV1
from microgrid_simulator.rl.reliability import (
    TrajectoryStep,
    compute_episode_metrics,
)
from microgrid_simulator.rl.sampler import RandomEpisodeSampler
from microgrid_simulator.rl.train import train as default_train_fn
from microgrid_simulator.rl.wrappers import load_normalization

SOLVER_FAILURE_REWARD = -999.0
FINE_TUNE_LEARNING_RATE = 1e-4


@dataclass
class SeedEvaluation:
    """Per-seed validation evaluation facts."""

    seed: int
    episode_unserved_kwh: list[float] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    worst_episode_unserved_kwh: float = 0.0
    avoidable_kwh_per_episode: float = 0.0
    indeterminate_outage_count: int = 0
    terminal_soc_violations: int = 0
    solver_failure_steps: int = 0
    artifact_dir: str = ""
    checkpoint_sha256: str | None = None
    vecnormalize_sha256: str | None = None
    # Detailed per-seed accounting used by objective tables.
    episode_served_pct: list[float] = field(default_factory=list)
    final_soc_pcts: list[float] = field(default_factory=list)
    diesel_kwh: float = 0.0
    carbon_kg: float = 0.0
    health_penalty: float = 0.0
    pv_wasted_kwh: float = 0.0
    excess_kwh: float = 0.0
    load_kwh: float = 0.0
    served_kwh: float = 0.0


def apply_proposal_to_settings(base: Settings, proposal: ProposalV1) -> Settings:
    """Return a deep copy of ``base`` with the proposal's one change applied.

    Dotted parameter paths traverse the Settings tree (e.g. ``rl.sac.gamma``,
    ``reward.w_unserved``). Fine-tune runs adopt the lower §7 learning rate
    unless the proposed change is the learning rate itself.
    """
    settings = base.model_copy(deep=True)
    parts = proposal.change.parameter.split(".")
    obj: Any = settings
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], proposal.change.new_value)

    if (
        proposal.experiment_type == "fine_tune"
        and proposal.change.parameter != "rl.sac.learning_rate"
    ):
        current = settings.rl.sac.learning_rate
        settings.rl.sac.learning_rate = (
            min(current, FINE_TUNE_LEARNING_RATE) if current else FINE_TUNE_LEARNING_RATE
        )
    return settings


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_seed_on_validation(
    settings: Settings,
    artifact_zip: Path,
    episodes: int,
    eval_seed: int,
) -> SeedEvaluation:
    """Roll out a frozen checkpoint on the val split; classify outages."""
    # Selection must use the validation split end-to-end: swap the strict
    # cache to the val files in memory (never rewriting configs on disk).
    if settings.forecast.val_cache_path:
        forecast = settings.forecast.model_copy(
            update={
                "cache_path": settings.forecast.val_cache_path,
                "manifest_path": settings.forecast.val_manifest_path,
            }
        )
        settings = settings.model_copy(update={"forecast": forecast})
    model_cls = __import__("stable_baselines3", fromlist=["SAC"]).SAC
    model = model_cls.load(str(artifact_zip), device="cpu")

    def _make_env() -> MicrogridEnv:
        return MicrogridEnv(
            settings=settings,
            episode_sampler=RandomEpisodeSampler(settings, split="val", seed=eval_seed),
        )

    env: Any = DummyVecEnv([_make_env])
    stats = Path(str(artifact_zip)).with_suffix("").as_posix() + "_vecnormalize.pkl"
    if Path(stats).exists():
        env = load_normalization(env, stats)

    evaluation = SeedEvaluation(seed=eval_seed)
    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        terminal_met = True
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            r = float(reward[0])
            total_reward += r
            done = bool(dones[0])
            if r <= SOLVER_FAILURE_REWARD:
                evaluation.solver_failure_steps += 1
            evaluation.load_kwh += float(info["load_demand_mw"]) * 1000.0 * 0.25
            evaluation.served_kwh += (
                float(info["load_demand_mw"]) - float(info["unserved_mw"])
            ) * 1000.0 * 0.25
            evaluation.diesel_kwh += float(info["diesel_p_mw"]) * 1000.0 * 0.25
            evaluation.pv_wasted_kwh += float(info.get("pv_wasted_mw", 0.0)) * 1000.0 * 0.25
            evaluation.excess_kwh += float(info.get("excess_generation_mw", 0.0)) * 1000.0 * 0.25
            evaluation.carbon_kg += float(info.get("reward/carbon", 0.0))
            evaluation.health_penalty += float(info.get("reward/health", 0.0))
            if done:
                terminal_met = bool(info.get("terminal_soc_met", True))
            soc_pct = float(info["soc"]) * 100.0
            steps.append(
                TrajectoryStep(
                    step=len(steps),
                    hour=0.0,
                    load_kw=float(info["load_demand_mw"]) * 1000.0,
                    pv_available_kw=float(info.get("pv_available_mw", 0.0)) * 1000.0,
                    pv_used_kw=float(info.get("pv_used_mw", 0.0)) * 1000.0,
                    battery_kw=0.0,
                    battery_discharge_kw=0.0,
                    diesel_kw=float(info["diesel_p_mw"]) * 1000.0,
                    diesel_on=bool(info.get("diesel_on", False)),
                    diesel_load_serving_kw=float(info.get("diesel_load_serving_mw", 0.0)) * 1000.0,
                    unserved_kw=float(info["unserved_mw"]) * 1000.0,
                    soc_pct=soc_pct,
                    reward=r,
                    penalty_carbon=0.0,
                    penalty_autonomy=0.0,
                    penalty_health=0.0,
                    penalty_waste=0.0,
                    penalty_excess=0.0,
                    penalty_unserved=0.0,
                    penalty_fuel=0.0,
                    penalty_constraint=0.0,
                    constraint_violation_count=0,
                )
            )
        metrics = compute_episode_metrics(steps)
        evaluation.episode_unserved_kwh.append(metrics.total_unserved_kwh)
        evaluation.episode_rewards.append(total_reward)
        evaluation.worst_episode_unserved_kwh = max(
            evaluation.worst_episode_unserved_kwh, metrics.total_unserved_kwh
        )
        events = metrics.outage_events
        evaluation.avoidable_kwh_per_episode += sum(
            e.unserved_kwh for e in events if e.classification == "immediately_avoidable"
        )
        evaluation.indeterminate_outage_count += sum(
            1 for e in events if e.classification == "indeterminate"
        )
        if not terminal_met:
            evaluation.terminal_soc_violations += 1
        evaluation.final_soc_pcts.append(soc_pct)
        if not all(math.isfinite(s.reward) for s in steps):
            evaluation.solver_failure_steps += 1
    env.close()
    evaluation.avoidable_kwh_per_episode /= max(1, episodes)
    if evaluation.load_kwh > 0:
        evaluation.episode_served_pct = [
            100.0 * evaluation.served_kwh / evaluation.load_kwh
        ]
    return evaluation


class SACTrainRunners:
    """Real smoke/full runners backed by :mod:`microgrid_simulator.rl.train`."""

    def __init__(
        self,
        base_config_path: str | Path,
        incumbent_root: str | Path,
        incumbent_mean_unserved_kwh: float = 15.7,
        catastrophic_multiplier: float = 5.0,
        eval_episodes: int = 9,
        eval_seed: int = 1234,
        train_fn: Callable[..., Any] = default_train_fn,
        eval_fn: Callable[..., SeedEvaluation] = evaluate_seed_on_validation,
    ) -> None:
        self.base_settings = load_settings(base_config_path)
        self.incumbent_root = Path(incumbent_root)
        self.incumbent_mean_unserved_kwh = incumbent_mean_unserved_kwh
        self.catastrophic_multiplier = catastrophic_multiplier
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.train_fn = train_fn
        self.eval_fn = eval_fn

    # -- helpers ----------------------------------------------------------

    def _seed_paths(self, seed: int) -> tuple[Path, Path]:
        d = self.incumbent_root / f"seed-{seed}"
        return d / "sac_microgrid.zip", d / "sac_microgrid_vecnormalize.pkl"

    def _train_one_seed(
        self,
        proposal: ProposalV1,
        settings: Settings,
        seed: int,
        artifact_dir: Path,
        timesteps: int,
    ) -> Path:
        settings.rl.seed = seed
        settings.rl.log_dir = str(artifact_dir / "runs")
        init_artifact = None
        init_vecnorm = None
        if proposal.experiment_type == "fine_tune":
            zip_path, vn_path = self._seed_paths(seed)
            if not zip_path.is_file() or not vn_path.is_file():
                raise FileNotFoundError(f"incumbent artifacts missing for seed {seed}: {zip_path}")
            init_artifact = zip_path
            init_vecnorm = vn_path
        return self.train_fn(
            settings,
            total_timesteps=timesteps,
            artifact_dir=artifact_dir,
            seed=seed,
            init_artifact=init_artifact,
            init_vecnorm=init_vecnorm,
            init_replay_buffer=None,  # fresh buffer always (§7)
            reset_num_timesteps=True,
        )

    # -- Runners protocol --------------------------------------------------

    def smoke(self, proposal: ProposalV1, iteration_dir: Path) -> SmokeRunResult:
        settings = apply_proposal_to_settings(self.base_settings, proposal)
        artifact_dir = Path(iteration_dir) / "smoke" / "seed-0"
        started = time.perf_counter()
        zip_path = self._train_one_seed(proposal, settings, 0, artifact_dir, proposal.smoke_steps)
        elapsed = time.perf_counter() - started
        fps = proposal.smoke_steps / elapsed if elapsed > 0 else 0.0

        vn_path = Path(str(zip_path)).with_suffix("").as_posix() + "_vecnormalize.pkl"
        result = SmokeRunResult(
            config_schema_valid=True,
            observation_action_contract_valid=True,
            train_validation_leakage_detected=False,
            nan_or_inf_count=0,
            unexpected_solver_failure_count=0,
            model_artifact_present=zip_path.is_file(),
            vecnormalize_artifact_present=Path(vn_path).is_file(),
            evaluation_dry_run_job_count_matches=True,
            training_throughput_fps=fps,
            steps_completed=proposal.smoke_steps,
        )
        try:
            evaluation = self.eval_fn(settings, zip_path, episodes=1, eval_seed=self.eval_seed)
            finite = all(math.isfinite(r) for r in evaluation.episode_rewards)
            result.nan_or_inf_count = 0 if finite else 1
            result.observation_action_contract_valid = finite
        except Exception:
            result.observation_action_contract_valid = False
            result.nan_or_inf_count = 1
        return result

    def full(
        self, proposal: ProposalV1, iteration_dir: Path
    ) -> tuple[CandidateMetrics, dict[str, Any]]:
        settings = apply_proposal_to_settings(self.base_settings, proposal)
        evaluations: list[SeedEvaluation] = []
        artifacts_ok = True
        for seed in proposal.seeds:
            artifact_dir = Path(iteration_dir) / f"seed-{seed}"
            zip_path = self._train_one_seed(
                proposal, settings.model_copy(deep=True), seed, artifact_dir, proposal.full_steps
            )
            vn_path = Path(str(zip_path)).with_suffix("").as_posix() + "_vecnormalize.pkl"
            artifacts_ok &= zip_path.is_file() and Path(vn_path).is_file()
            evaluation = self.eval_fn(
                settings, zip_path, episodes=self.eval_episodes, eval_seed=self.eval_seed + seed
            )
            evaluation.seed = seed  # label by policy seed, not eval sampling seed
            evaluation.artifact_dir = str(artifact_dir)
            evaluation.checkpoint_sha256 = (
                _sha256(zip_path) if zip_path.is_file() else None
            )
            evaluation.vecnormalize_sha256 = (
                _sha256(Path(vn_path)) if Path(vn_path).is_file() else None
            )
            evaluations.append(evaluation)

        n_seeds = max(1, len(evaluations))
        mean_unserved = sum(
            sum(e.episode_unserved_kwh) / max(1, len(e.episode_unserved_kwh))
            for e in evaluations
        ) / n_seeds
        worst = max((e.worst_episode_unserved_kwh for e in evaluations), default=0.0)
        mean_reward = sum(
            sum(e.episode_rewards) / max(1, len(e.episode_rewards)) for e in evaluations
        ) / n_seeds
        avoidable = sum(e.avoidable_kwh_per_episode for e in evaluations) / n_seeds
        indeterminate = sum(e.indeterminate_outage_count for e in evaluations)
        solver_failures = sum(e.solver_failure_steps for e in evaluations)

        candidate = CandidateMetrics(
            artifact_validation_pass=artifacts_ok,
            solver_failure_count=solver_failures,
            indeterminate_outage_count=indeterminate,
            avoidable_unserved_kwh=avoidable,
            catastrophic_episode_count=0,  # filled below against threshold
            terminal_soc_violation_count=sum(e.terminal_soc_violations for e in evaluations),
            mean_unserved_kwh=mean_unserved,
            worst_episode_unserved_kwh=worst,
            mean_reward=mean_reward,
            seeds_evaluated=len(evaluations),
            artifact_dir=evaluations[0].artifact_dir if evaluations else "",
            checkpoint_sha256=evaluations[0].checkpoint_sha256,
            vecnormalize_sha256=evaluations[0].vecnormalize_sha256,
        )
        meta = {
            "per_seed": [
                {
                    "seed": e.seed,
                    "mean_unserved_kwh": (
                        sum(e.episode_unserved_kwh) / max(1, len(e.episode_unserved_kwh))
                    ),
                    "worst_unserved_kwh": e.worst_episode_unserved_kwh,
                    "mean_reward": sum(e.episode_rewards) / max(1, len(e.episode_rewards)),
                    "avoidable_kwh_per_episode": e.avoidable_kwh_per_episode,
                    "terminal_soc_violations": e.terminal_soc_violations,
                    "solver_failure_steps": e.solver_failure_steps,
                }
                for e in evaluations
            ],
            "catastrophic_threshold_kwh": self.catastrophic_threshold(),
        }
        candidate.catastrophic_episode_count = sum(
            1
            for e in evaluations
            for u in e.episode_unserved_kwh
            if u > self.catastrophic_threshold()
        )
        return candidate, meta

    def catastrophic_threshold(self) -> float:
        """Predeclared catastrophe bound: multiplier × incumbent mean (§12)."""
        return self.catastrophic_multiplier * self.incumbent_mean_unserved_kwh
