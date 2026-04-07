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

