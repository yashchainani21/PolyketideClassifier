# Methylated Mixed-Extender OOD Evaluation Set

## Overview

This evaluation set tests whether a PKS classifier can recognize polyketide products that combine two challenges: (1) a novel extender unit in one extension module and (2) an alpha-carbon methylation simulating C-methyltransferase tailoring activity.

**Final size:** 15,413 unique molecules

## Construction Pipeline

### Step 1: Generate mixed-extender bound products (script 16)

2-extension-module PKS products were generated using `scripts/16_generate_mixed_extender_ood_set.py`, which:

- Loads all 11 extender codes (6 training + 5 held-out OOD)
- Partitions `structureDB` extension modules by AT substrate into training vs OOD sets
- Generates all 2-module combinations with exactly 1 training + 1 OOD module (both positional orderings)
- Crosses all 29 starter units with the mixed extension combos
- Releases bound products via thiolysis and cyclization
- Deduplicates and removes overlaps with training data

**Training extenders:** Malonyl-CoA, Methylmalonyl-CoA, allylmal, hmal, emal, mxmal
**Held-out (OOD) extenders:** butmal, hexmal, isobutmal, D-isobutmal, DCP

This produced 40,385 unique mixed-extender SMILES (`mixed_ood_pks_products_2_ext_no_stereo_1train_1ood_SMILES.txt`).

### Step 2: Filter to carboxylic acids

From the 40,385 mixed SMILES:
- Kept only molecules containing a carboxylic acid group (`[CX3](=O)[OX2H1]`)
- Excluded molecules with lactone rings (`[OX2;R][CX3](=O)`)
- 19,855 passed the filter (20,034 had no COOH, 496 had lactone rings)

### Step 3: Alpha-carbon methylation

For each carboxylic acid molecule:
- Identified the alpha carbon (carbon neighbor of the carbonyl C in -COOH)
- Added a methyl group via RWMol if the alpha carbon had at least 1 hydrogen
- Removed stereochemistry and canonicalized the SMILES

15,413 successfully methylated (4,442 skipped due to quaternary alpha carbon with no available H).

### Step 4: Deduplication and overlap removal

- Deduplicated methylated SMILES (15,413 already unique)
- Removed any overlap with training SMILES (0 overlapping molecules)
- All output SMILES validated as parseable with no stereochemistry markers

## Files

| File | Description |
|---|---|
| `eval_methylated_mixed_ood_pks_2_ext_no_stereo_SMILES.txt` | One SMILES per line, 15,413 molecules |
| `eval_methylated_mixed_ood_pks_2_ext_no_stereo.pkl` | Python pickle: `{smiles: smiles}` dict |

## Evaluation Results

Using supervised GNN (`best_model.pt`, epoch 10) and ECFP4 logistic regression probe:

| Metric | GNN | ECFP4 |
|---|---|---|
| Recall (>=0.5) | 45.3% | 4.5% |
| Mean PKS prob | 0.437 | 0.054 |
| Median PKS prob | 0.109 | 0.0003 |

Results JSON: `models/supervised_gnn/ood_eval_methylated_mixed_ood.json`

## Context: Comparison with other eval sets

| Eval Set | Molecules | GNN Recall | ECFP4 Recall |
|---|---|---|---|
| Extender-code OOD (pure novel extenders) | 678 | ~42-46% | ~46% |
| Methylated, training extenders, 1-ext | 394 | 99.7% | 30.2% |
| Methylated, training extenders, 1-2 ext | 9,815 | 99.9% | 6.5% |
| **Methylated, mixed extenders (this set)** | **15,413** | **45.3%** | **4.5%** |

The mixed methylated set combines two OOD perturbations. One OOD extender module is enough to drop GNN recall from 99.9% to 45.3%, while the added methylation causes ECFP4 to collapse to 4.5%.
