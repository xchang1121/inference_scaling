import os

from training.settings import load_settings

# Tokenizers fork-parallelism deadlocks under the trainers' worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
settings = load_settings()
for stage in settings["stages"]:
    print(f"== {stage}", flush=True)
    if stage == "download":
        from training.download import run

        run(settings)
    elif stage == "grpo":
        from training.grpo import run

        run(settings)
    elif stage == "vrpo_preferences":
        from training.vrpo import preferences

        preferences(settings)
    else:
        from training.vrpo import train

        train(settings)
