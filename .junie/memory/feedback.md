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

