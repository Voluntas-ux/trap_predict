# site

A single self-contained HTML file. No build step, no dependencies, no server: open `index.html` in a browser, or publish this directory with GitHub Pages (Settings → Pages → deploy from branch, folder `/site`).

## What it does

The page reimplements the round generators, the seven-family model library and the six-statistic battery in JavaScript, so the simulations run live rather than being read off a table:

- **The console** generates a round from any controller algorithm at any contamination strength, fires it target by target, and shows the board filling, the posterior over families shifting, the next-target predictive, and running accuracy against the counting baseline. The same round can then be passed through the predecessor battery for comparison.
- **The scheme-reuse demo** observes a range whose controller draws from a library of K whole-round schemes, and plots how quickly a scheme registry beats counting and how fast the evidence accumulates.
- **The static tables** are transcribed from the Python runs in `verification/logs/`.

## Agreement with the reference implementation

The JavaScript engine is a port, not an approximation. Checked against the Python: null moments of all six statistics to three decimals, log Bayes factors of 83.6 / 43.4 / 26.1 / 80.2 for A1 / A2 / A4 / A5 at λ = 1 in both, counting baseline 0.606 against the exact 0.6067, and mixture accuracy 0.770 against the Python's 0.768 on the same alternative.

If you change a likelihood in `src/trap_controller_inference.py`, change it here too — the port is in the `ENGINE` block at the top of the script section, and it is deliberately written to mirror the Python function by function.

## Editing

Colours, type and layout are set in the `:root` block. The three direction classes carry fixed colours throughout the page (left cyan, straight bone, right rose); keep that mapping if you add charts, since the board, the legends and the trajectory diagram all rely on it.
