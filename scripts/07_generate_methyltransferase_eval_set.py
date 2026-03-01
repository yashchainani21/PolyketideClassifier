"""
Methyltransferase-Modified PKS Evaluation Set Generation Script

This script generates an OOD evaluation set by simulating methyltransferase domain
activity on PKS products made from training extenders. The pipeline:

1. Generate bound PKS products with 1-2 extension modules (training extenders only)
2. Release via thiolysis only (carboxylic acid products)
3. Filter to carboxylic acids without lactone rings
4. Methylate the alpha carbon (simulating C-methyltransferase activity)
5. Deduplicate, remove training overlap, and save

This produces molecules sharing the same backbone as training PKS products but with
a subtle alpha-methylation — testing whether the model learned general PKS features
vs. overfitting to exact extender side-chain patterns.
"""

from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from itertools import product
from dataclasses import dataclass
import multiprocessing as mp
import pickle
import time
import glob
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import bcs


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_EXTENSION_MODULES = 2

# Starter units (None = use all available)
STARTER_CODES: Optional[List[str]] = None

# Training extenders only (same as script 01)
EXTENDER_CODES: List[str] = [
    'Malonyl-CoA', 'Methylmalonyl-CoA', 'allylmal', 'hmal', 'emal', 'mxmal'
]

REMOVE_STEREOCHEMISTRY = True

# Parallelization settings
NUM_WORKERS: Optional[int] = None  # None = use all CPUs
CHUNKSIZE_BOUND = 1000
CHUNKSIZE_UNBOUND = 500

# Output settings
OUTPUT_DIR = "../data/processed"

# Trivial products to filter out
TRIVIAL_PRODUCTS = {'S', 'O=C=O'}

# SMARTS patterns for filtering
CARBOXYLIC_ACID_SMARTS = '[CX3](=O)[OX2H1]'
LACTONE_RING_SMARTS = '[OX2;R][CX3](=O)'


# =============================================================================
# PHASE 1: BOUND PRODUCT GENERATION (reused from script 05)
# =============================================================================

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """Modify the starter and extender acyl-CoA units used to generate PKS products."""
    if starter_codes is not None:
        for key in list(bcs.starters.keys()):
            if key not in starter_codes:
                bcs.starters.pop(key, None)

    if extender_codes is not None:
        for key in list(bcs.extenders.keys()):
            if key not in extender_codes:
                bcs.extenders.pop(key, None)


def build_bcs_cluster_and_product(starter: str, extension_mods_combo):
    """Build a bcs PKS cluster and product given a starter and extension module combo."""
    try:
        loading_AT_domain = bcs.AT(active=True, substrate=starter)
        loading_module = bcs.Module(domains=OrderedDict({bcs.AT: loading_AT_domain}), loading=True)
        full_modules = [loading_module] + list(extension_mods_combo)
        cluster = bcs.Cluster(modules=full_modules)
        product_mol = cluster.computeProduct(structureDB)
        return cluster, product_mol
    except Exception as e:
        print(f"Error building loading module with starter {starter} combo {extension_mods_combo}: {e}")
        return None, None


def generate_all_bound_products() -> List[Tuple]:
    """Generate all bound PKS products for 1-2 extension modules."""
    all_cluster_product_pairs = []

    with mp.Pool() as pool:
        for i in range(1, MAX_EXTENSION_MODULES + 1):
            print(f"\nGenerating clusters and products with {i} extension module(s)...")

            starter_plus_ext_mods_combos = product(
                bcs.starters.keys(),
                product(extension_modules_list, repeat=i)
            )

            results_i = pool.starmap(
                build_bcs_cluster_and_product,
                starter_plus_ext_mods_combos,
                chunksize=CHUNKSIZE_BOUND
            )

            results_i = [r for r in results_i if None not in r]
            all_cluster_product_pairs.extend(results_i)
            print(f"  Generated {len(results_i)} products with {i} extension module(s)")

    return all_cluster_product_pairs


# =============================================================================
# PHASE 2: THIOLYSIS RELEASE (reused from script 05, thiolysis only)
# =============================================================================

@dataclass
class UnboundProductResult:
    """Container for results from processing a single bound PKS molecule."""
    original_index: int
    thiolysis_smiles: List[str]
    thiolysis_error: Optional[str] = None


def run_thiolysis(bound_product_mol: Chem.Mol) -> List[Chem.Mol]:
    """Run thiolysis reaction to release a bound PKS product as carboxylic acid."""
    Chem.SanitizeMol(bound_product_mol)
    rxn = AllChem.ReactionFromSmarts(
        '[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]'
    )
    products = rxn.RunReactants((bound_product_mol,))
    if not products:
        raise ValueError("Unable to perform thiolysis reaction")

    unbound_products = []
    for prod_tuple in products:
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                unbound_products.append(prod)
            except:
                continue
    return unbound_products


def process_single_bound_product(args: Tuple[int, bytes, bool]) -> UnboundProductResult:
    """Process one bound PKS molecule through thiolysis only."""
    original_index, pickled_data, remove_stereochemistry = args
    _, bound_pks_mol = pickle.loads(pickled_data)

    thiolysis_smiles: List[str] = []
    thiolysis_error: Optional[str] = None

    try:
        unbound_products = run_thiolysis(bound_pks_mol)
        for unbound_mol in unbound_products:
            if remove_stereochemistry:
                Chem.RemoveStereochemistry(unbound_mol)
            smiles = Chem.MolToSmiles(unbound_mol)
            if smiles not in TRIVIAL_PRODUCTS:
                thiolysis_smiles.append(smiles)
    except Exception as e:
        thiolysis_error = str(e)

    return UnboundProductResult(
        original_index=original_index,
        thiolysis_smiles=thiolysis_smiles,
        thiolysis_error=thiolysis_error,
    )


def prepare_work_items(bound_pks_products: List[Tuple],
                       remove_stereochemistry: bool) -> List[Tuple[int, bytes, bool]]:
    """Pre-pickle inputs with their original indices for passing to workers."""
    work_items = []
    for i, (pks_design, bound_mol) in enumerate(bound_pks_products):
        pickled_data = pickle.dumps((pks_design, bound_mol))
        work_items.append((i, pickled_data, remove_stereochemistry))
    return work_items


def process_all_unbound_products(bound_pks_products: List[Tuple]) -> List[UnboundProductResult]:
    """Process all bound products through thiolysis in parallel."""
    print("\nPreparing work items for thiolysis...")
    prep_start = time.time()
    work_items = prepare_work_items(bound_pks_products, REMOVE_STEREOCHEMISTRY)
    prep_time = time.time() - prep_start
    print(f"Work items prepared in {prep_time:.2f} seconds")

    num_workers = NUM_WORKERS or mp.cpu_count()
    total_items = len(work_items)

    print(f"\nStarting parallel thiolysis with {num_workers} workers...")
    print(f"Total items to process: {total_items}")

    results: List[UnboundProductResult] = []
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_items, desc="Thiolysis", unit="mol") as pbar:
            for result in pool.imap_unordered(process_single_bound_product,
                                              work_items,
                                              chunksize=CHUNKSIZE_UNBOUND):
                results.append(result)
                pbar.update(1)

    return results


def collect_and_deduplicate_thiolysis(results: List[UnboundProductResult]) -> List[str]:
    """Collect and deduplicate thiolysis SMILES."""
    sorted_results = sorted(results, key=lambda r: r.original_index)
    unique_smiles: set = set()
    smiles_list: List[str] = []

    for result in sorted_results:
        for smiles in result.thiolysis_smiles:
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                smiles_list.append(smiles)

    return smiles_list


# =============================================================================
# PHASE 3: FILTER TO CARBOXYLIC ACIDS (no lactone rings)
# =============================================================================

def filter_carboxylic_acids(smiles_list: List[str]) -> List[str]:
    """Keep only molecules with a carboxylic acid group and no lactone rings."""
    carboxylic_acid_pat = Chem.MolFromSmarts(CARBOXYLIC_ACID_SMARTS)
    lactone_ring_pat = Chem.MolFromSmarts(LACTONE_RING_SMARTS)

    filtered = []
    no_cooh = 0
    has_lactone = 0
    parse_fail = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            parse_fail += 1
            continue
        if not mol.HasSubstructMatch(carboxylic_acid_pat):
            no_cooh += 1
            continue
        if mol.HasSubstructMatch(lactone_ring_pat):
            has_lactone += 1
            continue
        filtered.append(smi)

    print(f"  Parse failures: {parse_fail}")
    print(f"  No carboxylic acid: {no_cooh}")
    print(f"  Has lactone ring: {has_lactone}")
    print(f"  Passed filter: {len(filtered)}")

    return filtered


# =============================================================================
# PHASE 4: ALPHA-CARBON METHYLATION
# =============================================================================

def methylate_alpha_carbon(smiles: str) -> Optional[str]:
    """
    Add a methyl group to the alpha carbon of a carboxylic acid.

    The alpha carbon is the carbon neighbor of the carbonyl C in -COOH.
    Returns None if alpha carbon has no available hydrogen (quaternary).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    carboxylic_acid_pat = Chem.MolFromSmarts(CARBOXYLIC_ACID_SMARTS)
    match = mol.GetSubstructMatch(carboxylic_acid_pat)
    if not match:
        return None

    carbonyl_c_idx = match[0]

    # Find alpha carbon: carbon neighbor of carbonyl C (not oxygen)
    alpha_c_idx = None
    for neighbor in mol.GetAtomWithIdx(carbonyl_c_idx).GetNeighbors():
        if neighbor.GetAtomicNum() == 6:  # carbon, not oxygen
            alpha_c_idx = neighbor.GetIdx()
            break

    if alpha_c_idx is None:
        return None

    # Check alpha carbon has at least 1 hydrogen
    if mol.GetAtomWithIdx(alpha_c_idx).GetTotalNumHs() < 1:
        return None

    # Add methyl group using RWMol
    rwmol = Chem.RWMol(mol)
    methyl_c_idx = rwmol.AddAtom(Chem.Atom(6))
    rwmol.AddBond(alpha_c_idx, methyl_c_idx, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rwmol)
    except Exception:
        return None

    # Remove stereochemistry and canonicalize
    Chem.RemoveStereochemistry(rwmol)
    return Chem.MolToSmiles(rwmol)


def methylate_all(smiles_list: List[str]) -> Tuple[List[str], int, int]:
    """
    Methylate all carboxylic acid SMILES at their alpha carbon.

    Returns
    -------
    Tuple[List[str], int, int]
        (methylated_smiles, num_skipped_no_alpha_h, num_skipped_error)
    """
    methylated = []
    skipped_no_alpha_h = 0
    skipped_error = 0

    for smi in tqdm(smiles_list, desc="Methylating", unit="mol"):
        result = methylate_alpha_carbon(smi)
        if result is None:
            # Distinguish between no-alpha-H and other errors
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                pat = Chem.MolFromSmarts(CARBOXYLIC_ACID_SMARTS)
                match = mol.GetSubstructMatch(pat)
                if match:
                    carbonyl_c_idx = match[0]
                    has_alpha_c = False
                    for neighbor in mol.GetAtomWithIdx(carbonyl_c_idx).GetNeighbors():
                        if neighbor.GetAtomicNum() == 6:
                            has_alpha_c = True
                            if neighbor.GetTotalNumHs() < 1:
                                skipped_no_alpha_h += 1
                            else:
                                skipped_error += 1
                            break
                    if not has_alpha_c:
                        skipped_no_alpha_h += 1
                else:
                    skipped_error += 1
            else:
                skipped_error += 1
        else:
            methylated.append(result)

    return methylated, skipped_no_alpha_h, skipped_error


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point for methyltransferase eval set generation."""
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_filename = "eval_methylated_pks_2_ext_no_stereo"

    print("=" * 70)
    print("METHYLTRANSFERASE-MODIFIED PKS EVALUATION SET GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Max extension modules: {MAX_EXTENSION_MODULES}")
    print(f"  Starter codes: {STARTER_CODES or 'All available'}")
    print(f"  Extender codes: {EXTENDER_CODES}")
    print(f"  Remove stereochemistry: {REMOVE_STEREOCHEMISTRY}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # =========================================================================
    # PHASE 1: Generate Bound PKS Products
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Generating Bound PKS Products (1-2 extension modules)")
    print("=" * 70)

    phase1_start = time.time()
    bound_pks_products = generate_all_bound_products()
    phase1_time = time.time() - phase1_start

    print(f"\nGenerated {len(bound_pks_products)} bound PKS products")
    print(f"Phase 1 completed in {phase1_time:.2f} seconds")

    # =========================================================================
    # PHASE 2: Thiolysis Release
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Releasing via Thiolysis (carboxylic acids only)")
    print("=" * 70)

    phase2_start = time.time()
    results = process_all_unbound_products(bound_pks_products)
    phase2_time = time.time() - phase2_start

    thiolysis_errors = sum(1 for r in results if r.thiolysis_error)
    if thiolysis_errors:
        print(f"\nEncountered {thiolysis_errors} thiolysis errors")

    thiolysis_smiles = collect_and_deduplicate_thiolysis(results)
    print(f"\n{len(thiolysis_smiles)} unique SMILES after thiolysis + dedup")
    print(f"Phase 2 completed in {phase2_time:.2f} seconds")

    # =========================================================================
    # PHASE 3: Filter to Carboxylic Acids (no lactones)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Filtering to Carboxylic Acids (no lactone rings)")
    print("=" * 70)

    phase3_start = time.time()
    filtered_smiles = filter_carboxylic_acids(thiolysis_smiles)
    phase3_time = time.time() - phase3_start

    print(f"Phase 3 completed in {phase3_time:.2f} seconds")

    # =========================================================================
    # PHASE 4: Methylate Alpha Carbon
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Methylating Alpha Carbon")
    print("=" * 70)

    phase4_start = time.time()
    methylated_smiles, skipped_no_h, skipped_error = methylate_all(filtered_smiles)
    phase4_time = time.time() - phase4_start

    print(f"\n  Successfully methylated: {len(methylated_smiles)}")
    print(f"  Skipped (no H on alpha C): {skipped_no_h}")
    print(f"  Skipped (other error): {skipped_error}")
    print(f"Phase 4 completed in {phase4_time:.2f} seconds")

    # =========================================================================
    # PHASE 5: Post-processing and Output
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Deduplication, Overlap Removal, and Output")
    print("=" * 70)

    phase5_start = time.time()

    # Deduplicate methylated SMILES
    pre_dedup = len(methylated_smiles)
    seen = set()
    unique_methylated = []
    for smi in methylated_smiles:
        if smi not in seen:
            seen.add(smi)
            unique_methylated.append(smi)
    print(f"\nDeduplicated: {pre_dedup} -> {len(unique_methylated)} unique SMILES")

    # Remove overlap with training SMILES
    train_smiles_files = glob.glob(os.path.join(OUTPUT_DIR, "pks_products_*_SMILES.txt"))
    if train_smiles_files:
        train_smiles: set = set()
        for fpath in train_smiles_files:
            with open(fpath) as f:
                train_smiles.update(line.strip() for line in f if line.strip())
        print(f"Loaded {len(train_smiles)} training SMILES from {len(train_smiles_files)} file(s)")

        pre_overlap = len(unique_methylated)
        unique_methylated = [smi for smi in unique_methylated if smi not in train_smiles]
        overlap_count = pre_overlap - len(unique_methylated)
        print(f"Removed {overlap_count} SMILES overlapping with training data")
    else:
        print("WARNING: No training SMILES files found — skipping overlap removal")

    # Validate all output SMILES
    invalid = sum(1 for smi in unique_methylated if Chem.MolFromSmiles(smi) is None)
    if invalid:
        print(f"WARNING: {invalid} invalid SMILES in output!")
    stereo_count = sum(1 for smi in unique_methylated if '@' in smi or '/' in smi or '\\' in smi)
    if stereo_count:
        print(f"WARNING: {stereo_count} SMILES still contain stereochemistry markers!")

    # Save as SMILES->SMILES dict (identity mapping, consistent with script 05)
    output_dict = {smi: smi for smi in unique_methylated}

    output_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}.pkl")
    print(f"\nSaving to: {output_filepath}")
    with open(output_filepath, "wb") as f:
        pickle.dump(output_dict, f)

    smiles_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_SMILES.txt")
    print(f"Saving SMILES to: {smiles_filepath}")
    with open(smiles_filepath, "w") as f:
        for smi in unique_methylated:
            f.write(smi + "\n")

    phase5_time = time.time() - phase5_start

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Bound products generated:     {len(bound_pks_products)}")
    print(f"  After thiolysis + dedup:      {len(thiolysis_smiles)}")
    print(f"  After COOH filter:            {len(filtered_smiles)}")
    print(f"  After methylation:            {len(methylated_smiles)}")
    print(f"  After final dedup:            {pre_dedup} -> {len(unique_methylated) + overlap_count if train_smiles_files else len(unique_methylated)}")
    print(f"  After overlap removal:        {len(unique_methylated)}")
    print(f"\n  Final eval set size:          {len(unique_methylated)}")
    print(f"\nTiming:")
    print(f"  Phase 1 (bound products):     {phase1_time:.2f}s")
    print(f"  Phase 2 (thiolysis):          {phase2_time:.2f}s")
    print(f"  Phase 3 (COOH filter):        {phase3_time:.2f}s")
    print(f"  Phase 4 (methylation):        {phase4_time:.2f}s")
    print(f"  Phase 5 (output):             {phase5_time:.2f}s")
    print(f"  Total:                        {total_time:.2f}s")
    print(f"\nOutput files:")
    print(f"  {output_filepath}")
    print(f"  {smiles_filepath}")
    print("\nDone!")


# =============================================================================
# SCRIPT INITIALIZATION
# =============================================================================

# Modify bcs starters and extenders before importing retrotide
modify_bcs_starters_extenders(starter_codes=STARTER_CODES, extender_codes=EXTENDER_CODES)
print(f"\nNumber of starter units: {len(bcs.starters)}")
print(f"Number of extender units: {len(bcs.extenders)}\n")

# Import retrotide and structureDB only after modifying bcs.starters and bcs.extenders
from retrotide import retrotide, structureDB
print(f"Number of entries in structureDB: {len(structureDB)}\n")

# Get list of extension modules
extension_modules_list = list(structureDB.keys())


if __name__ == "__main__":
    main()
