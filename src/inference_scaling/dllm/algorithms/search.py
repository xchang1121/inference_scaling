"""Finite-budget search and trajectory sharpening for block-diffusion LMs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


def _exact_logprob(sample: DiffusionSample) -> float:
    if sample.trajectory_logprob is None:
        raise RuntimeError("the backend omitted an exact reverse-trajectory probability")
    return float(sample.trajectory_logprob)


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHStep:
    update: int
    proposal: DiffusionSample
    previous_trajectory_logprob: float
    acceptance_probability: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHResult:
    prompt: TokenSequence
    alpha: float
    initial: DiffusionSample
    steps: tuple[DiffusionPowerMHStep, ...]
    final: DiffusionSample

    @property
    def acceptance_rate(self) -> float:
        return sum(step.accepted for step in self.steps) / len(self.steps)


def run_diffusion_trajectory_power_mh(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionPowerMHConfig,
    sampling: DiffusionSamplingConfig,
    seed: int = 0,
) -> DiffusionPowerMHResult:
    """Target ``q(trace | prompt)^alpha`` with independence proposals from ``q``.

    SDAR exposes the probability of a committed reverse trajectory, while the
    marginal probability of the final token sequence is generally intractable.
    This is therefore the precise block-diffusion analogue of probability
    sharpening: its state is the sampled trajectory, including the final text.
    """

    if not sampling.has_exact_trajectory_density:
        raise ValueError("trajectory-power MH requires an exact diffusion policy")
    sampling.validate_generation_length(config.total_length)
    seeds = SeedStream(seed)
    requests = [
        DiffusionGenerationRequest(
            prefix=prompt,
            generation_length=config.total_length,
            sampling=sampling,
            seed=seeds.derive("dllm-power-mh", "proposal", draw),
            request_id=f"dllm-power-mh:proposal:{draw}",
        )
        for draw in range(config.updates + 1)
    ]
    samples = backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of MH proposals")
    for sample in samples:
        _exact_logprob(sample)

    current = samples[0]
    initial = current
    steps: list[DiffusionPowerMHStep] = []
    for update, proposal in enumerate(samples[1:], start=1):
        previous_logprob = _exact_logprob(current)
        proposal_logprob = _exact_logprob(proposal)
        log_acceptance = min(
            0.0,
            (config.alpha - 1.0) * (proposal_logprob - previous_logprob),
        )
        acceptance_probability = exp(log_acceptance)
        uniform = float(seeds.generator("dllm-power-mh", "accept", update).random())
        accepted = log(max(uniform, float.fromhex("0x1.0p-1022"))) <= log_acceptance
        if accepted:
            current = proposal
        steps.append(
            DiffusionPowerMHStep(
                update=update,
                proposal=proposal,
                previous_trajectory_logprob=previous_logprob,
                acceptance_probability=acceptance_probability,
                accepted=accepted,
            )
        )
    return DiffusionPowerMHResult(
        prompt=prompt,
        alpha=config.alpha,
        initial=initial,
        steps=tuple(steps),
        final=current,
    )


@dataclass(frozen=True, slots=True)
class DiffusionBeamHypothesis:
    token_ids: TokenSequence
    trajectory_logprob: float
    samples: tuple[DiffusionSample, ...]


@dataclass(frozen=True, slots=True)
class DiffusionBlockBeamStage:
    generated_length_before: int
    proposals: int
    retained: tuple[DiffusionBeamHypothesis, ...]


@dataclass(frozen=True, slots=True)
class DiffusionBlockBeamResult:
    prompt: TokenSequence
    stages: tuple[DiffusionBlockBeamStage, ...]
    beams: tuple[DiffusionBeamHypothesis, ...]

    @property
    def best(self) -> DiffusionBeamHypothesis:
        return self.beams[0]


def run_diffusion_block_beam(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionBlockBeamConfig,
    sampling: DiffusionSamplingConfig,
    seed: int = 0,
) -> DiffusionBlockBeamResult:
    """Retain the highest-density sampled continuations at each block boundary.

    The first stage draws ``width`` blocks from the root.  Later stages draw
    ``branching_factor`` continuations per retained beam.  This is sampled
    block search, not vocabulary-enumerating token beam search.
    """

    if not sampling.has_exact_trajectory_density:
        raise ValueError("block beam search requires an exact diffusion policy")
    sampling.validate_generation_length(config.decision_block_size)
    seeds = SeedStream(seed)
    beams = (DiffusionBeamHypothesis((), 0.0, ()),)
    stages: list[DiffusionBlockBeamStage] = []
    stage_count = config.total_length // config.decision_block_size
    for stage_index in range(stage_count):
        requests: list[DiffusionGenerationRequest] = []
        owners: list[int] = []
        draws_per_beam = config.width if stage_index == 0 else config.branching_factor
        for beam_index, beam in enumerate(beams):
            for draw_index in range(draws_per_beam):
                requests.append(
                    DiffusionGenerationRequest(
                        prefix=prompt + beam.token_ids,
                        generation_length=config.decision_block_size,
                        sampling=sampling,
                        seed=seeds.derive(
                            "dllm-block-beam", stage_index, beam_index, draw_index
                        ),
                        request_id=(
                            f"dllm-block-beam:stage:{stage_index}:beam:{beam_index}:"
                            f"draw:{draw_index}"
                        ),
                    )
                )
                owners.append(beam_index)
        sampled = backend.sample_batch(requests)
        if len(sampled) != len(requests):
            raise RuntimeError("backend returned an invalid number of beam proposals")
        expanded = [
            DiffusionBeamHypothesis(
                token_ids=beams[owner].token_ids + sample.token_ids,
                trajectory_logprob=(
                    beams[owner].trajectory_logprob + _exact_logprob(sample)
                ),
                samples=beams[owner].samples + (sample,),
            )
            for owner, sample in zip(owners, sampled, strict=True)
        ]
        expanded.sort(
            key=lambda item: (item.trajectory_logprob, item.token_ids),
            reverse=True,
        )
        beams = tuple(expanded[: config.width])
        stages.append(
            DiffusionBlockBeamStage(
                generated_length_before=stage_index * config.decision_block_size,
                proposals=len(expanded),
                retained=beams,
            )
        )
    return DiffusionBlockBeamResult(prompt=prompt, stages=tuple(stages), beams=beams)


__all__ = [
    "DiffusionBeamHypothesis",
    "DiffusionBlockBeamResult",
    "DiffusionBlockBeamStage",
    "DiffusionPowerMHResult",
    "DiffusionPowerMHStep",
    "run_diffusion_block_beam",
    "run_diffusion_trajectory_power_mh",
]
