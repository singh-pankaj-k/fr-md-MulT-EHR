[2026-04-07 10:37] - Updated by Junie
{
    "TYPE": "preference",
    "CATEGORY": "default dataset",
    "EXPECTATION": "Use MIMIC-IV as the default dataset across all scripts, configs, and examples.",
    "NEW INSTRUCTION": "WHEN dataset is not explicitly specified THEN default to MIMIC-IV everywhere"
}

[2026-04-07 10:47] - Updated by Junie
{
    "TYPE": "correction",
    "CATEGORY": "dependency install",
    "EXPECTATION": "Accommodate macOS where DGL is unavailable by adjusting dependencies or offering an alternative path.",
    "NEW INSTRUCTION": "WHEN platform is macOS THEN omit DGL from requirements and document an alternative backend"
}

[2026-04-07 13:06] - Updated by Junie
{
    "TYPE": "correction",
    "CATEGORY": "dev mode handling",
    "EXPECTATION": "Pipeline should always run with dev=False; only dataset differs between dev and full training.",
    "NEW INSTRUCTION": "WHEN configuring any pipeline stage THEN force dev=False in code and configs"
}

[2026-04-07 14:40] - Updated by Junie
{
    "TYPE": "correction",
    "CATEGORY": "model head dimension",
    "EXPECTATION": "Fix the root cause of the drug_rec output-target size mismatch by aligning the model output dimension with the dataset’s label space (no silencing).",
    "NEW INSTRUCTION": "WHEN creating task-specific heads THEN set num_classes from dataset label mapping"
}

