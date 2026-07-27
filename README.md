# Blood Supply Chain Simulator & Hierarchical Extended Blood Matching for Sickle Cell Patients (WSC 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19543714.svg)](https://doi.org/10.5281/zenodo.19543714)

This repository contains the software implementation and source code for our paper titled **"Simulating a Hierarchical Supply Chain with Extended Blood Type Matching for Sickle Cell Patients"** (Proceedings of the 2026 Winter Simulation Conference, WSC).

> F. B. Oyebolu, A. Tehim, M. Chion, S. Trompeter, N. Gleadall and W. J. Astle, "Simulating a Hierarchical Supply Chain with Extended Blood Type Matching for Sickle Cell Patients," Proceedings of the 2026 Winter Simulation Conference (WSC), 2026.

This work extends earlier modelling from:
- WSC 2024 implementation repository: [`HDSci/BloodMatcher-WSC24`](https://github.com/HDSci/BloodMatcher-WSC24)
- Core simulator lineage: [`floyebolu/BSC-Simulator`](https://github.com/floyebolu/BSC-Simulator)

---

## Contents & Structure of the Repository

The repository contains:

1. **A discrete-time simulation model** for evaluating blood matching rules in a **multi-echelon hierarchical blood supply chain** for transfusion-dependent sickle cell patients.  
   This can be found in the [`BSCSimulator`](BSCSimulator) package.

2. **A Google OR-Tools-based network solver** for daily minimum-cost flow allocation of red blood cell (RBC) units to requests and balancing flows.  
   This can be found in [`mincostflow.pyx`](BSCSimulator/mincostflow.pyx).

3. **Experiment definitions** for comparing matching and logistics policies under different transhipment assumptions and transportation penalties.  
   - Experiment structure is in [`experiments`](BSCSimulator/experiments).  
   - Scenario execution/definitions are centered in experiment scripts (for example, files analogous to `allo_incidence.py` in prior repositories).

4. **Policy evaluation and tuning components** for penalty-weight sensitivity and trade-off analysis (clinical risk vs transport burden).  
   - Bayesian optimisation and/or parameter exploration components are in the experiment/tuning modules where applicable.

5. **Analysis scripts and notebooks** for reproducing figures/tables and summarising simulation outputs.  
   - Plotting/table helper functions are in `analysis/scripts`.
   - Notebooks for manuscript analysis are in `analysis/notebooks/`.

---

## Installation

This software was developed primarily on Linux systems with Python 3.11.  
Dependencies are listed in [`requirements.txt`](requirements.txt).

However, please note:
- We cannot guarantee out-of-the-box compatibility on native Windows, especially for Cython/C++ compilation. We recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/about).
- If using notebooks, they may require a separate environment depending on your workflow and optional analysis dependencies.

### Cython Extension(s)

After installing dependencies, compile the Cython extension:

```bash
python setup.py build_ext --inplace
```

This builds the OR-Tools-backed minimum-cost flow module used in daily matching/allocation.

---

## Execution

### Experiments

Run the simulator using:

```bash
python -m BSCSimulator.main [parameterfile]
```

If no parameter file is provided, default parameters from the repository defaults are used (typically in `BSCSimulator/experiments/default_parameters.yml`).

If optional parameter files are used:
- provided values override defaults;
- dictionary-style parameters will require full dictionary replacement (recursive override behaviour is not supported).

Output is written to top-level `out/`, including:
- experiment outputs (date-stamped subdirectories),
- logs in `out/logs/`.

### Analysis

Analysis notebooks/scripts are used to:
- compute manuscript metrics (e.g., expected alloimmunisation , shortages, expiries, transhipments),
- generate scenario comparison tables and figures,
- inspect SHU-level workload and movement patterns.

If notebook paths are environment-specific, update relative paths or run notebooks from repository root.

---

## Scenarios Represented in the WSC 2026 Study

The 2026 manuscript evaluates policies in the hierarchical NHSBT network with SHU transhipments:

- **L01**: Limited matching; siloed SHUs (stock balancing-only transhipments)
- **E01**: Extended matching; siloed SHUs
- **E02**: Extended matching; unrestricted transhipments without transport penalty
- **E03**: Extended matching; transhipments penalized by transportation cost term

---

## General Notes & Assumptions

### Antigens

Antigens are encoded as bit-fields in integer representations, including major and extended antigens:
- Major: A, B, D
- Extended/minor set used in matching policies includes: C, c, E, e, K, k, Fya, Fyb, Jka, Jkb, M, N, S, s

### Alloantibodies

Alloantibody-related structures are handled as profile/mask-like representations in simulation state and compatibility logic.  
As in earlier code lineage, ensure mask/profile interpretation is consistent when reading precomputed data and when checking compatibility in matching routines.

### Demand

Requests are represented with structured integer-array records (e.g., request identifier, phenotype, units, due date, group, and location identifiers), with exact schema depending on experiment generation mode.

### Supply

Donor units are represented with structured integer-array records (e.g., unit identifier, antigen phenotype, collection date, location identifiers), then tracked through inventory, transit, allocation, and expiry pipelines.

### Locations and Movements

The hierarchical supply chain representation includes:
- manufacturing-region links,
- SHU-level stock,
- transhipments for balancing and patient-specific matching,
- movement tracking for workload and logistics metrics.

---

## WSC 2026 Model Context (Summary)

This repository corresponds to experiments analysing a **three-tier hierarchical blood supply chain** with bilateral SHU transhipments and extended RBC matching for SCD patients.  
Key trade-off explored:

- stricter focus on compatibility can reduce alloimmunisation risk significantly;
- unrestricted cross-network transhipments can create major operational burden;
- introducing a transportation penalty enables practical trade-offs between clinical gains and logistics workload.

---

## Relationship to Other Repositories

- [`floyebolu/BSC-Simulator`](https://github.com/floyebolu/BSC-Simulator): actively developed simulator framework and broader feature set.
- [`HDSci/BloodMatcher-WSC24`](https://github.com/HDSci/BloodMatcher-WSC24): code snapshot used for the WSC 2024 publication.
- `HDSci/BloodMatcher-WSC2026` (this repository): code snapshot and experiment setup corresponding to the WSC 2026 manuscript.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this software in your research, please cite the corresponding WSC 2026 paper.

```bibtex
@inproceedings{oyebolu2026wsc_bloodmatcher,
  author    = {Oyebolu, Folarin B. and Tehim, Anisha and Chion, Marie and Trompeter, Sara and Gleadall, Nicholas and Astle, William J.},
  title     = {Simulating a Hierarchical Supply Chain with Extended Blood Type Matching for Sickle Cell Patients},
  booktitle = {Proceedings of the 2026 Winter Simulation Conference (WSC)},
  year      = {2026}
}
```

If you also build on the earlier model formulation, please additionally cite the WSC 2024 paper:
- Oyebolu et al., *Optimization of Extended Red Blood Cell Matching in Transfusion Dependent Sickle Cell Patients*, WSC 2024, DOI: [10.1109/WSC63780.2024.10838863](https://doi.org/10.1109/WSC63780.2024.10838863)
