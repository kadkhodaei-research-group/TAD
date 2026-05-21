# TAD

**Temperature-Aware Dimer (TAD)**: A saddle point search method designed for temperature-dependent energy surfaces that account for vibrational effects. Ideal for strongly anharmonic solids with low-temperature mechanical (phonon) instability, where density functional theory forces are unsuitable for dimer searches.

<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc-sa.png" alt="Creative Commons License" width="75"> [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

This code has been developed by **Seyyedfaridoddin Fattahpour** and **Sara Kadkhodaei**.

## Reference / Citation

Please cite the following works if you use any part of this code, workflow, or data:

1. **Seyyedfaridoddin Fattahpour, Sara Kadkhodaei** (2025). *Diffusion Transition States in Temperature-Dependent Solids with Strong Anharmonic Vibrations: A Dual-GPR-Dimer Approach*. Under preparation.

2. **Seyyedfaridoddin Fattahpour, Sara Kadkhodaei** (2023). Improving ab initio diffusion calculations in materials through Gaussian process regression. *Physical Review Materials* **8**, 013804. [DOI: 10.1103/PhysRevMaterials.8.013804](https://doi.org/10.1103/PhysRevMaterials.8.013804)

3. **Seyyedfaridoddin Fattahpour et al.** (2022). Understanding the role of anharmonic phonons in diffusion of bcc metals. *Physical Review Materials* **6**, 023803. [DOI: 10.1103/PhysRevMaterials.6.023803](https://doi.org/10.1103/PhysRevMaterials.6.023803)

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. See the `LICENSE` file for the full legal text.

You may share and adapt this material for non-commercial purposes, provided that you give appropriate credit, link to the license, indicate if changes were made, and distribute any adapted material under the same license.

License deed: [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## `run_studies_template.sh` — generic temperature-study launcher

Generic replacement for `run_studies_screen*.sh`. Every choice that varies
between studies (system, mode, temperatures, study sizes, …) lives in a
separate config file; the launcher itself never has to be edited.

---

## Files

| File | Purpose |
|---|---|
| `run_studies_template.sh` | The launcher. Reads a config file and dispatches `run_tad.py` for every (system, study, temperature, repetition). |
| `studies.conf.template` | Empty skeleton config. Copy to `studies.conf` and fill in for a new system. |
| `studies.conf.example` | Filled config that reproduces `run_studies_screen.sh` (Zr + Mo). |
| `studies.conf.example_zirconia` | Filled config that reproduces both `run_studies_screen_zirconia.sh` (VASP) and `run_studies_screen_zirconia_eam.sh` (EAM). |

The launcher always reads `./studies.conf` next to itself unless you pass
`--config <path>`.

---

## Quick start

```bash
cd /path/to/scripts
cp studies.conf.example studies.conf      # or studies.conf.example_zirconia
./run_studies_template.sh
```

---

## Reproducing the three legacy launchers

### Reproduce `run_studies_screen.sh` (Zr + Mo)

```bash
cp studies.conf.example studies.conf
./run_studies_template.sh
```

Filters mirror the legacy script:

| Legacy invocation | Template invocation |
|---|---|
| `./run_studies_screen.sh` | `./run_studies_template.sh` |
| `./run_studies_screen.sh --system zr` | `./run_studies_template.sh --system zr` |
| `./run_studies_screen.sh --system mo` | `./run_studies_template.sh --system mo` |
| `./run_studies_screen.sh --study 100/100` | `./run_studies_template.sh --study 100snapshots_100repititions` |
| `./run_studies_screen.sh --study 50/10` | `./run_studies_template.sh --study 50snapshots_10repititions` |
| `./run_studies_screen.sh --system zr --study 50/10` | `./run_studies_template.sh --system zr --study 50snapshots_10repititions` |

(The template uses the full study name because that string is also the
output-directory prefix.)

### Reproduce `run_studies_screen_zirconia.sh` (ZrO2, VASP)

```bash
cp studies.conf.example_zirconia studies.conf
./run_studies_template.sh --system zro2_vasp
```

### Reproduce `run_studies_screen_zirconia_eam.sh` (ZrO2, EAM)

```bash
cp studies.conf.example_zirconia studies.conf
./run_studies_template.sh --system zro2_eam
```

### Run both ZrO2 variants in one go

```bash
cp studies.conf.example_zirconia studies.conf
./run_studies_template.sh
```

---

## Run-time filtering

```bash
./run_studies_template.sh                      # everything in the config
./run_studies_template.sh --system NAME        # only that system
./run_studies_template.sh --study  NAME        # only that named study
./run_studies_template.sh --system NAME --study NAME
./run_studies_template.sh --config <path>      # alternate config file
./run_studies_template.sh --help
```

`NAME` for `--system` is one of the entries in the config's `SYSTEMS` array;
`NAME` for `--study` is the part before the first `:` in a `STUDIES` entry.

---

## Adding a new system

1. `cp studies.conf.template studies.conf`
2. Add your system name to `SYSTEMS=( … )`.
3. Add a `SYS_<NAME>_*` block. Required fields: `POSCAR`, `FC_PATTERN`,
   `EXECUTION_MODE`, `TEMPERATURES`. Optional: `EAM_POTENTIAL_FILE`,
   `KIM_MODEL`, `VASP_COMMAND`, `VASP_INPUT_DIR`, `MOVING_INDICES`,
   `ORIENT_ATOM_DIRECTION`, `HYPERPARAMS`, `PARALLEL_RUNS`, `STUDIES`,
   `OUTPUT_LABEL`.
4. `./run_studies_template.sh`

`SYS_<NAME>_POSCAR` and `SYS_<NAME>_FC_PATTERN` may contain `{TEMP}` (replaced
with the current temperature) and shell globs (`*`, `?`, `[…]`).
`MOVING_INDICES` and `ORIENT_ATOM_DIRECTION` can be left empty to let
`run_tad.py` auto-detect.

The two example configs are good models to copy from.

---

## Troubleshooting

- **`ERROR: Config file not found: …`** — copy one of the example configs to
  `studies.conf`, or pass `--config <path>`.
- **`ERROR: SYS_<NAME>_<FIELD> is not set in <config>`** — you added a system
  to `SYSTEMS` but forgot one of its required fields.
- **`ERROR: System '<NAME>' has no studies …`** — neither global `STUDIES`
  nor `SYS_<NAME>_STUDIES` is non-empty for that system.
- **`ERROR: System '<NAME>' has no hyperparameters …`** — neither global
  `HYPERPARAMS` nor `SYS_<NAME>_HYPERPARAMS` is non-empty for that system.
- **`ERROR: <kind> pattern matched no files: …`** — your POSCAR or FC pattern
  contains a glob that didn't match anything. Double-check the path and the
  `{TEMP}` substitution.
- **A run dies inside `run_tad.py`** — each (system, study, temperature, rep)
  has its own log next to its output directory: `<output-dir>.log`.
