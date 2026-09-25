"""Finite-budget search and trajectory sharpening for block-diffusion LMs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from inference_scaling.dllm.algorithms.config import (
    DiffusionBlockBeamConfig,
    DiffusionPowerMHConfig,
)
from inference_scaling.dllm.config import DiffusionSamplingConfig, diffusion_decision_stage_lengths
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
)
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


def _exact_logprob(sample: DiffusionSample) -> float:
    if sample.trajectory_logprob is None:
        raise RuntimeError("the backend omitted an exact reverse-trajectory probability")
    return float(sample.trajectory_logprob)


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHStep:
    stage_length: int
    update: int
    cut: int
    previous_base_trajectory_logprob: float
    proposed_base_trajectory_logprob: float
    previous_proposal_trajectory_logprob: float
    proposed_proposal_trajectory_logprob: float
    log_acceptance: float
    acceptance_probability: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHBlock:
    sample: DiffusionSample
    base_trajectory_logprob: float
    proposal_trajectory_logprob: float

    @property
    def token_ids(self) -> TokenSequence:
        return self.sample.token_ids


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHState:
    prompt: TokenSequence
    blocks: tuple[DiffusionPowerMHBlock, ...]

    @property
    def token_ids(self) -> TokenSequence:
        return tuple(token for block in self.blocks for token in block.token_ids)

    @property
    def base_trajectory_logprob(self) -> float:
        return sum(block.base_trajectory_logprob for block in self.blocks)

    @property
    def proposal_trajectory_logprob(self) -> float:
        return sum(block.proposal_trajectory_logprob for block in self.blocks)


@dataclass(frozen=True, slots=True)
class DiffusionPowerMHResult:
    prompt: TokenSequence
    alpha: float
    initial: DiffusionPowerMHState
    steps: tuple[DiffusionPowerMHStep, ...]
    final: DiffusionPowerMHState

    @property
    def acceptance_rate(self) -> float:
        return sum(step.accepted for step in self.steps) / len(self.steps)


def _sample_power_suffix(
    *,
    backend: DiffusionBackend,
    prefix: TokenSequence,
    length: int,
    base_sampling: DiffusionSamplingConfig,
    proposal_sampling: DiffusionSamplingConfig,
    seeds: SeedStream,
    key: tuple[object, ...],
    request_id: str,
) -> tuple[DiffusionPowerMHBlock, ...]:
    """Draw a suffix one native block at a time.

    Each block's canvas ends at the block, so its base and proposal
    probabilities come from the same logits and stay valid when a later cut
    regenerates the blocks after it.
    """

    blocks: list[DiffusionPowerMHBlock] = []
    tokens: TokenSequence = ()
    for index in range(length // proposal_sampling.block_length):
        sample = backend.sample_batch([DiffusionGenerationRequest(
            prefix=prefix + tokens, generation_length=proposal_sampling.block_length, sampling=proposal_sampling,
            seed=seeds.derive(*key, index), request_id=f"{request_id}:block:{index}",
            reference_temperature=base_sampling.temperature,
        )])[0]
        if sample.reference_trajectory_logprob is None:
            raise RuntimeError("the backend omitted the base trajectory probability")
        blocks.append(DiffusionPowerMHBlock(sample, float(sample.reference_trajectory_logprob), _exact_logprob(sample)))
        tokens += sample.token_ids
    return tuple(blocks)


def run_diffusion_trajectory_power_mh(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionPowerMHConfig,
    sampling: DiffusionSamplingConfig,
    proposal_sampling: DiffusionSamplingConfig | None = None,
    seed: int = 0,
) -> DiffusionPowerMHResult:
    """Target ``p(trace | prompt)^alpha`` by aligned suffix resampling.

    The reverse trajectory has a tractable probability, while the marginal
    probability of the final token sequence is generally intractable. Cuts are
    restricted to complete diffusion blocks, so the forward and reverse suffix
    proposal probabilities are both exact. Staged growth and a tempered
    proposal mirror the ARLLM experiment without splitting a jointly denoised
    block.
    """

    if not sampling.has_exact_trajectory_density:
        raise ValueError("trajectory-power MH requires an exact diffusion policy")
    if sampling.top_k or sampling.top_p < 1:
        raise ValueError("trajectory-power MH requires a full-support base policy")
    proposal_sampling = proposal_sampling or replace(
        sampling,
        temperature=sampling.temperature / config.alpha,
    )
    if not proposal_sampling.has_exact_trajectory_density:
        raise ValueError("trajectory-power MH requires an exact proposal policy")
    if proposal_sampling.top_k or proposal_sampling.top_p < 1:
        raise ValueError("trajectory-power MH requires a full-support proposal")
    schedule = (
        sampling.block_length,
        sampling.steps_per_block,
        sampling.remasking,
    )
    proposal_schedule = (
        proposal_sampling.block_length,
        proposal_sampling.steps_per_block,
        proposal_sampling.remasking,
    )
    if schedule != proposal_schedule:
        raise ValueError("base and proposal trajectory schedules must match")
    stage_lengths = diffusion_decision_stage_lengths(
        total_length=config.total_length,
        decision_block_size=config.decision_block_size,
        sampling=sampling,
    )
    seeds = SeedStream(seed)
    current = DiffusionPowerMHState(prompt, ())
    initial: DiffusionPowerMHState | None = None
    steps: list[DiffusionPowerMHStep] = []
    stage_length = 0
    global_update = 0
    for stage_index, extension_length in enumerate(stage_lengths):
        extension_blocks = _sample_power_suffix(
            backend=backend,
            prefix=prompt + current.token_ids,
            length=extension_length,
            base_sampling=sampling,
            proposal_sampling=proposal_sampling,
            seeds=seeds,
            key=("dllm-power-mh", stage_index, "extend"),
            request_id=f"dllm-power-mh:stage:{stage_index}:extend",
        )
        current = DiffusionPowerMHState(prompt, current.blocks + extension_blocks)
        stage_length += extension_length
        if initial is None:
            initial = current
        for stage_update in range(config.updates_per_stage):
            global_update += 1
            cut_block = int(
                seeds.generator(
                    "dllm-power-mh", stage_index, stage_update, "cut"
                ).integers(0, len(current.blocks))
            )
            kept_blocks = current.blocks[:cut_block]
            old_blocks = current.blocks[cut_block:]
            cut = sum(len(block.token_ids) for block in kept_blocks)
            suffix_length = stage_length - cut
            proposed_blocks = _sample_power_suffix(
                backend=backend,
                prefix=prompt + current.token_ids[:cut],
                length=suffix_length,
                base_sampling=sampling,
                proposal_sampling=proposal_sampling,
                seeds=seeds,
                key=("dllm-power-mh", stage_index, stage_update, "proposal"),
                request_id=f"dllm-power-mh:stage:{stage_index}:update:{stage_update}",
            )
            old_p = sum(block.base_trajectory_logprob for block in old_blocks)
            old_q = sum(block.proposal_trajectory_logprob for block in old_blocks)
            new_p = sum(block.base_trajectory_logprob for block in proposed_blocks)
            new_q = sum(block.proposal_trajectory_logprob for block in proposed_blocks)
            uniform = float(
                seeds.generator(
                    "dllm-power-mh", stage_index, stage_update, "accept"
                ).random()
            )
            decision = decide_metropolis_hastings(
                current_target_log_density=config.alpha * old_p,
                proposed_target_log_density=config.alpha * new_p,
                forward_proposal_log_probability=new_q,
                reverse_proposal_log_probability=old_q,
                uniform=uniform,
            )
            log_acceptance = decision.log_acceptance
            acceptance_probability = decision.acceptance_probability
            accepted = decision.accepted
            if accepted:
                current = DiffusionPowerMHState(prompt, kept_blocks + proposed_blocks)
            steps.append(
                DiffusionPowerMHStep(
                    stage_length=stage_length,
                    update=global_update,
                    cut=cut,
                    previous_base_trajectory_logprob=old_p,
                    proposed_base_trajectory_logprob=new_p,
                    previous_proposal_trajectory_logprob=old_q,
                    proposed_proposal_trajectory_logprob=new_q,
                    log_acceptance=log_acceptance,
                    acceptance_probability=acceptance_probability,
                    accepted=accepted,
                )
            )
    assert initial is not None
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
    seeds = SeedStream(seed)
    beams = (DiffusionBeamHypothesis((), 0.0, ()),)
    stages: list[DiffusionBlockBeamStage] = []
    stage_lengths = diffusion_decision_stage_lengths(
        total_length=config.total_length,
        decision_block_size=config.decision_block_size,
        sampling=sampling,
    )
    generated_length = 0
    for stage_index, stage_length in enumerate(stage_lengths):
        requests: list[DiffusionGenerationRequest] = []
        owners: list[int] = []
        draws_per_beam = config.width if stage_index == 0 else config.branching_factor
        for beam_index, beam in enumerate(beams):
            for draw_index in range(draws_per_beam):
                requests.append(
                    DiffusionGenerationRequest(
                        prefix=prompt + beam.token_ids,
                        generation_length=stage_length,
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
                generated_length_before=generated_length,
                proposals=len(expanded),
                retained=beams,
            )
        )
        generated_length += stage_length
    return DiffusionBlockBeamResult(prompt=prompt, stages=tuple(stages), beams=beams)


__all__ = [
    "DiffusionBeamHypothesis",
    "DiffusionBlockBeamResult",
    "DiffusionBlockBeamStage",
    "DiffusionPowerMHBlock",
    "DiffusionPowerMHResult",
    "DiffusionPowerMHState",
    "DiffusionPowerMHStep",
    "run_diffusion_block_beam",
    "run_diffusion_trajectory_power_mh",
]
