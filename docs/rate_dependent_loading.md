# Rate-dependent loading extension

The repository now includes a first rate-dependent extension of the supported
T4 + T3 tensegrity model.

## Modeling choice

The tensegrity structure itself is **not** replaced by springs. It retains the
unilateral member law already used elsewhere in the repository:

- cables carry tension only;
- bars/struts carry compression only.

Rate dependence is introduced in the **external basement/foundation
attachment**. The attachment is represented by a Kelvin-Voigt element,

\[
F_b = -k_f u_b - c_f \dot u_b,
\]

where `k_f` is the attachment stiffness and `c_f` is the attachment damping.

At each time step the structure is equilibrated using a backward-Euler
incremental potential. This gives a quasi-static tensegrity network coupled to
a rate-dependent basement attachment; it is not yet a full inertial tissue
model.

## New modules

- `src/tensegrity/loading.py` provides linear and smooth loading histories.
- `src/tensegrity/viscoelastic.py` provides the Kelvin-Voigt basement model and
  its backward-Euler incremental potential.
- `experiments/t4_rate_sweep.py` compares slow, medium, and fast force ramps.
- `tests/test_loading.py` and `tests/test_viscoelastic.py` validate the new
  mechanics utilities.

## Current demonstration

The experiment uses normalized/model units and intentionally avoids assigning
biological units before tissue data are available. The default run uses:

- T4 prestress scale: `0.40`
- basement stiffness: `1.0`
- basement damping: `0.25`
- final force per loaded node: `1e-4`
- ramp times: `0.1`, `1.0`, and `10.0`

Run:

```bash
PYTHONPATH=src:. python experiments/t4_rate_sweep.py
```

Outputs are written to `outputs/t4_rate_sweep/` and include the full time
history plus force-displacement, displacement-history, and basement-reaction
plots.

## Interpretation limits

The present parameters are dimensionless / normalized and are intended as a
numerical mechanics benchmark. They should not be interpreted as salamander
tissue properties. A later calibration step should use measured stress-strain
curves, loading rates, tissue geometry, and basement-attachment properties.
