> ## ⚠️ HOLD — do not publish before T3
>
> **Decision (2026-10-08):** this repository is **not pushed** and **not linked** in the first
> contact email. It is held together with the research note it belongs to
> (`14_ResearchNote_判定是抽样不是测量_EN_2026-10-08.md`).
>
> Reason: this README contains the same finding as that note. The two go out together or not at all.
> Publishing one without the other makes the email's framing incoherent.
>
> Full sequencing: `17_披露时序_发信前发信后_2026-10-08.md` in the application workspace.

**Language / 语言：** [English](#en) ｜ [中文](#zh)

---

<a id="en"></a>

# A Verdict Is a Sample, Not a Measurement

[English](README.md#en) | [中文](#zh)

## Abstract

This repository is a ~400-line demonstration that a model-as-judge verdict is a sampling procedure, plus a gate that stops the sampling from being reported as a measurement.

Two byte-identical copies of one image were each judged three times by a local vision-language model. The first copy returned `yes, partial, partial`, the second `yes, partial, yes`. Aggregated naively that is 90/100 and 100/100 for the same file. On this item the judge's decision distribution sits close to 50/50, so a 2:1 majority carries almost no information, and averaging over more runs does not repair that — it only makes the coin flip quieter.

The gate here requires the majority to reach 0.8 before a verdict is emitted; below that the instrument reports `uncertain`. In a closed loop it is what separated a real change from a 30-point artifact.

The finding is not new. It is established for text judges at n = 50. What is here is a first readout in the visual regime at n = 3–5, and a second failure mode in the cheap screening tier.

## Related Work

Schroeder & Wood-Doughty (2024), *Can You Trust LLM Judgments?* (arXiv 2412.12509), measure judge reliability with McDonald's omega and state in the abstract that fixed randomness does not guarantee it. The point that fixing the seed is not a fix is theirs, not mine.

Yagubyan (2026), *The Coin Flip Judge?* (arXiv 2606.13685), runs 50 pairwise and 50 pointwise repetitions per question across 29 tasks, measures a mean pairwise flip rate of 13.6%, and reports that a majority vote needs roughly 11 repetitions to recover the 50-trial verdict. Same claim, measured properly, at n = 50.

Gates are not new either. Calibrated abstention (arXiv 2608.07517) and SCOPE (arXiv 2602.13110, conformal) formalise the idea, and do it more rigorously than a hand-set threshold.

What I did not find is this characterisation applied to a vision judge. The multimodal-judge literature I located — MLLM-as-a-Judge (2024), MJ-Bench (NeurIPS 2025), Prometheus-Vision (2024), EvalMuse-40K (AAAI 2026) — measures capability and human-preference alignment, not run-to-run stability on identical inputs. Yagubyan lists cross-provider replication as an open next step.

Search coverage, stated plainly: OpenAlex, Crossref and DOAJ returned HTTP 200; arXiv was reached only indirectly; CNKI and Wanfang were not covered; my similarity matching degraded to surface form. A prior work phrased in entirely different words may have been missed.

## Contributions

1. A readout in the visual regime at n = 3–5. Not the phenomenon, and not the first observation of it — the first measurement I could find on identical visual inputs at this sample size.
2. A gate at 0.8 agreement. It is deliberately fail-closed, so it reports `uncertain` often, which is the safer failure and also the slower one.
3. A second failure mode in the cheap tier. MAD measures how much changed, not whether the change was right. With an empty prompt against a simple reference it reported 96.5 for a condition that asked for no change at all.
4. The instrument: `mad.py`, `verdict.py`, `judge.py`, `demo.py`, with a `--mock` flag so the aggregation logic can be read without a model. That mode is a logic demo, not evidence.

## Method

`judge.py` is the expensive tier — a pluggable judge, with `OllamaJudge` as the reference implementation against `gemma3:4b` and `qwen2.5vl:7b`. `mad.py` is the cheap tier: mean absolute pixel difference at 384×384, deterministic. `demo.py` copies the image under judgment to a second file, proves with MAD that the two are byte-identical, asks the same question of both copies `--repeat` times, and prints the verdicts with the gated aggregate.

```python
AGREE_MIN = 0.8

def aggregate(verdicts, agree_min=AGREE_MIN):
    top, n = Counter(verdicts).most_common(1)[0]
    frac = n / len(verdicts)
    if frac < agree_min:
        return None, frac, False      # None == "uncertain"
    return top, frac, True
```

At n = 3 only a unanimous 3:0 passes. At n = 5, 5:0 and 4:1 pass and 3:2 does not.

The gate earned its place in a closed loop: round 1 scored 70/100 and round 2 scored 100/100, and the whole 30-point swing came from one attribute whose verdicts went from `[partial, no, no]` to `[partial, yes, yes]` — 67% both times. Without the gate the conclusion would have been that the fix worked.

```bash
pip install -r requirements.txt

# needs a local vision model; gemma3:4b and qwen2.5vl:7b both work
ollama pull gemma3:4b

# compare an output against a reference (the setting the finding came from)
python demo.py output.png --reference reference.png

# or ask a single-image attribute question
python demo.py any_image.png
```

Do not expect the demo to reproduce the table in the Abstract, and it will say so. On a clean image with a capable-enough model every run agrees, and the tool prints that the item is nowhere near the judge's decision boundary instead of manufacturing a disagreement. Easy items are stable; the item in the Abstract was a fine-grained identity comparison, which is exactly what a small vision model is bad at. The recorded verdicts are the evidence and the demo is the instrument.

## Limitations

- n is 3–5 per condition, and one condition in the original work had n = 2. Nothing here is a rate estimate. The text-judge literature puts the requirement for a stable majority vote at about 11 repetitions, so a gate at n = 3 is deliberately fail-closed: it reports `uncertain` often, which is safe and low-throughput. That tension is not resolved here.
- The judge is deliberately weak, a 4B/27B local VLM. Some of the instability belongs to that judge rather than to model-as-judge in general. A stronger judge is more stable and is still a sampling procedure, so it still needs the gate.
- Instability is a boundary phenomenon, not a uniform noise floor. A clearly wrong case, a different garment, was stable 5/5. A borderline case was stable 1/5. Coarse verdicts hold up; fine ones do not.
- Fixing the seed does not help. With a fixed seed the judge returned 100/100/100, which freezes one sample rather than sampling the distribution. Repeats need different seeds.
- MAD is five lines and the judge is off-the-shelf. Neither is a contribution.

This is not a benchmark, a leaderboard, or a comparison of judges or aggregation rules, and the rates here should not be read as generalising beyond this instrument.

## License

MIT.

---

<a id="zh"></a>

# 判定是一次抽样，不是一次测量

[English](#en) | 中文

## 摘要

这个仓库是一个约 400 行的演示，用来说明模型当裁判给出的判定是一个抽样过程，并附上一个闸门，防止把抽样结果当成测量值上报。

把同一张图复制成两份字节完全相同的文件，各让一个本地视觉语言模型判定三次。第一份返回 `yes, partial, partial`，第二份返回 `yes, partial, yes`。直接汇总，同一个文件得到 90/100 和 100/100。这个条目上裁判的判定分布接近 50/50，所以 2:1 的多数票几乎没有信息量；多跑几次取平均修不了这件事，只是让抛硬币变得更安静。

这里给出的闸门要求多数票达到 0.8 才输出判定，否则输出 `uncertain`。在一个闭环里，是它把真实的变化和一个 30 分的假象分开了。

这个发现不是新的。它在 n = 50 的文本裁判中已经确立。这里给出的是视觉场景下 n = 3–5 的一次读数，以及便宜那一层的第二个失效模式。

## 相关工作

Schroeder & Wood-Doughty（2024），*Can You Trust LLM Judgments?*（arXiv 2412.12509），用 McDonald's omega 衡量裁判可靠性，并在摘要中说明固定随机性并不保证可靠。「固定种子不是修复手段」这个论断是他们的，不是我的。

Yagubyan（2026），*The Coin Flip Judge?*（arXiv 2606.13685），每题跑 50 次成对加 50 次单点评测，共 29 个任务，测得平均成对翻转率 13.6%，并报告多数票需要约 11 次重复才能恢复 50 次试验给出的判定。同一个论断，在 n = 50 下被正经量了出来。

闸门也不是新东西。Calibrated abstention（arXiv 2608.07517）和 SCOPE（arXiv 2602.13110，conformal）把这件事形式化了，而且比一个人手设定的阈值做得更严格。

我没有找到的，是把同一套刻画用在视觉裁判上。我检索到的多模态裁判文献——MLLM-as-a-Judge（2024）、MJ-Bench（NeurIPS 2025）、Prometheus-Vision（2024）、EvalMuse-40K（AAAI 2026）——衡量的是能力和人类偏好一致性，不是相同输入在多次运行间的稳定性。Yagubyan 也把跨 provider 复现列为开放的下一步。

检索覆盖范围说清楚：OpenAlex、Crossref、DOAJ 都返回 HTTP 200；arXiv 只是间接访问到的；CNKI 和万方没有覆盖；我的相似度匹配退化到了表层形式。用完全不同的措辞表述的前作可能被漏掉。

## 贡献

1. 视觉场景下 n = 3–5 的一次读数。不是这个现象，也不是对它的第一次观察，而是我没能找到的、在这个样本量下对相同视觉输入做过的一次测量。
2. 一个 0.8 一致度的闸门。它刻意偏向保守，所以会经常报 `uncertain`，这是更安全的失败方式，也是更慢的一种。
3. 便宜那一层的第二个失效模式。MAD 衡量的是改变了多少，不是改变得对不对。空提示词配一张简单参考图时，它为一个没有要求任何改变的条件报出 96.5。
4. 仪器本身：`mad.py`、`verdict.py`、`judge.py`、`demo.py`，并带 `--mock`，可以在没有模型的情况下读汇总逻辑。那个模式是逻辑演示，不是证据。

## 方法

`judge.py` 是贵的那一层——可插拔的裁判，参考实现是 `OllamaJudge`，跑 `gemma3:4b` 和 `qwen2.5vl:7b`。`mad.py` 是便宜的那一层：384×384 的平均绝对像素差，确定性计算。`demo.py` 会先把待判定的图复制成第二个文件，用 MAD 证明两份字节完全相同，然后就同一个问题问这两个副本 `--repeat` 次，打印判定结果和加了闸门的汇总值。

```python
AGREE_MIN = 0.8

def aggregate(verdicts, agree_min=AGREE_MIN):
    top, n = Counter(verdicts).most_common(1)[0]
    frac = n / len(verdicts)
    if frac < agree_min:
        return None, frac, False      # None == "uncertain"
    return top, frac, True
```

n = 3 时只有 3:0 全票通过。n = 5 时，5:0 和 4:1 通过，3:2 不通过。

这个闸门是在一个闭环里挣到位置的：第一轮 70/100，第二轮 100/100，30 分的全部摆动来自一个属性，它的判定从 `[partial, no, no]` 变成 `[partial, yes, yes]`，两次都是 67%。没有闸门，结论就会是修复生效了。

```bash
pip install -r requirements.txt

# 需要一个本地视觉模型；gemma3:4b 和 qwen2.5vl:7b 都可以
ollama pull gemma3:4b

# 把一个输出与参考图对比（发现来源的那个场景）
python demo.py output.png --reference reference.png

# 或者问一个单图属性问题
python demo.py any_image.png
```

不要指望这个 demo 复现摘要里那张表，它也会这么说。在一张干净的图上配一个足够强的模型，每次运行都会一致，工具会打印这个条目离裁判的决策边界很远，而不是硬造一个分歧。简单的条目是稳定的；摘要里那个条目是一次细粒度的身份比对，而这恰好是小型视觉模型不擅长的事。记录下来的判定是证据，demo 是仪器。

## 局限

- 每个条件下 n 是 3–5，原始工作里有一个条件是 n = 2。这里没有任何东西是比率估计。文本裁判文献把多数票稳定下来的要求放在约 11 次重复，所以 n = 3 的闸门是刻意偏向保守的：它经常报 `uncertain`，这安全，但吞吐低。这个张力在这里没有解决。
- 裁判是刻意用弱的，一个 4B/27B 的本地 VLM。一部分不稳定性属于那个裁判，不属于「模型当裁判」这件事。更强的裁判更稳定，但它仍然是抽样过程，仍然需要闸门。
- 不稳定性是边界现象，不是均匀的噪声底。一个明显错误的案例，换了件衣服，5/5 稳定。一个边界案例 1/5 稳定。粗粒度判定站得住，细粒度判定站不住。
- 固定种子没有用。固定种子下裁判返回 100/100/100，那只是冻结一个样本，而不是对分布采样。重复必须用不同的种子。
- MAD 是五行代码，裁判是现成的。两者都不是贡献。

这不是 benchmark、排行榜，也不是对裁判或汇总规则的比较；这里的比率不能读成可以泛化到这个仪器之外。

## 许可证

MIT。
