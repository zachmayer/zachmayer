# Densing Law of LLMs
#ai #scaling-laws #efficiency #paper

Source: https://www.nature.com/articles/s42256-025-01137-0?utm_source=substack&utm_medium=email
Date: 2026-08-29

Xiao, Cai, Zhao, Lin, Zeng, Zhou, Zheng, Han, Liu, Sun — Tsinghua / OpenBMB (the MiniCPM group). *Nature Machine Intelligence* 7, 1823–1833, Nov 2025. Open access, CC-BY. PDF: https://www.nature.com/articles/s42256-025-01137-0.pdf

## One-liner

Define **capability density** = effective params / actual params, where effective params is what a reference model would need to match the target's benchmark score. Across 51 open-weight base models since Llama-1, the max density doubles every **~3.5 months** (A ≈ 0.007/day, R² = 0.934). Same-performance model size halves on that cadence.

## Method (the part worth understanding)

Density can't be score/params because the relationship is nonlinear. Two-step fit:
1. **Loss scaling**: train 0.005B–0.8B reference models on the MiniCPM corpus at {10–60}×N tokens; fit conditional loss `L = aN^-α + bD^-β` where L = −log P(answer | instruction) on the *test set itself* (GPT-4o-generated reasoning traces as the answer text).
2. **Loss → score**: sigmoid `S = c / (1 + e^{-γ(L−l)}) + d`, fit on well-trained MiniCPM checkpoints (0.36B–4B). Held-out 10B+ models land on the curve.
3. Invert both at D₀ = 5T tokens to get effective N; density = effective / actual.

Benchmarks: MMLU, BBH, MATH, HumanEval, MBPP (few-shot, CoT). Base models only, no instruct tunes. Reference models and bench on HF: `openbmb/DensingLaw-ScalingBench`, `openbmb/DensingLaw-ScalingModels`.

## Findings

- **Doubling ~3.5 months** on public benches; **MMLU-CF** (contamination-free, Dec 2024) gives A = 0.0065 vs 0.0066 — the trend survives decontamination (R² = 0.953).
- **Post-ChatGPT acceleration**: slope 0.0048 before, 0.0073 after (+50%).
- **API price for GPT-3.5-level halves every ~2.6 months** — faster than density because inference infra (FlashAttention, PagedAttention, sparsity) compounds on top. GPT-3.5 $20/M (Dec 2022) → Gemini-1.5-Flash $0.075/M (Aug 2024), 266×.
- **Densing × Moore**: fixed-price chip compute doubles every 2.1y (Epoch); combined, the largest effective model runnable on a fixed-price chip doubles every **~88 days**.
- **Bigger ≠ denser**: Llama-3.1-405B is SOTA but low density; big runs are under-optimized relative to compute.
- **Compression usually lowers density**: Llama-3.2-1B/3B, Minitron-4B, and GPTQ variants all score below their parents. Only Gemma-2-9B (distilled from 27B) is denser than its source. Authors attribute it to under-training after prune/distill.
- Density gain 2023–25 came almost entirely from **data scale + quality** (Llama-1 1.4T → Llama-3 15T), not architecture or training algorithm. They expect MoE and pretraining-RL to matter next.

## Caveats (mine)

- Max-density envelope on ~51 points; the fit is on the frontier, not the population. Single-lab reference models (MiniCPM corpus) anchor everything.
- Fitting loss on the test set with GPT-4o rationales is clever but not obviously clean; the base-model-only restriction also excludes the models people actually use.
- Data ends April 2025 (Gemma-3, Llama-4-Scout, Mistral-Small-3.1). Extrapolating 3.5-month doubling through 2026 is a bet, not a measurement — they explicitly flag information-theoretic saturation and the need to keep refreshing benchmarks.
- They also define an inference-time density for MoE/quantized models (k₁ ≈ 2.45k₂ on an RTX 4090) but don't use it in the trend analysis.

## Why it matters for me

- Cost-planning heuristic: whatever a capability costs today, budget for ~2× cheaper per quarter on the open-weight side. Useful for the DDG local/on-device model conversation.
- Reinforces the [Open-Source LLM Parity Forecast](Open-Source%20LLM%20Parity%20Forecast.md) framing: the gap closes from below on a fixed cadence.
- Pairs with [WikiSkill](WikiSkill%20-%20Compiling%20Agent%20Experience%20into%20Persistent%20Knowledge.md): evolved skills let a 9B beat a bare 27B, which is a different lever on the same "capability per parameter" axis.

## Related

- [AI](../hubs/AI.md) — hub
- [Scaling Laws Carefully - Lilian Weng](Scaling%20Laws%20Carefully%20-%20Lilian%20Weng.md) — why power-law fits are fragile; read alongside this
- [Open-Source LLM Parity Forecast](Open-Source%20LLM%20Parity%20Forecast.md) — open-weight catch-up dates
- [WikiSkill - Compiling Agent Experience into Persistent Knowledge](WikiSkill%20-%20Compiling%20Agent%20Experience%20into%20Persistent%20Knowledge.md) — skills as the other capability-per-parameter lever
