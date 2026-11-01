"""Builds docs/getting_started.ipynb as an executed notebook.

Run from the package root:  python docs/build_vignette.py
"""

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
c = []  # cells

c.append(nbf.v4.new_markdown_cell(
    "# Getting started with `grangercm`\n\n"
    "This vignette walks through the full workflow: preparing series for\n"
    "stationarity, building a **Granger Causality Matrix**, correcting for\n"
    "multiple testing, visualising the result as a heatmap, and running the\n"
    "nonlinear backend.\n\n"
    "We use a synthetic 4-variable system with a *known* causal structure so\n"
    "you can check the answers:\n\n"
    "```\n"
    "x : exogenous noise\n"
    "y : driven by x at lag 1 (linear)\n"
    "w : driven by x at lag 2 through a nonlinear (tanh) channel\n"
    "u : independent noise\n"
    "```\n"
))

c.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n\n"
    "rng = np.random.default_rng(0)\n"
    "n = 600\n"
    "x = rng.normal(size=n)\n"
    "y = np.zeros(n)\n"
    "w = np.zeros(n)\n"
    "for t in range(2, n):\n"
    "    y[t] = 0.6 * x[t - 1] + 0.2 * y[t - 1] + rng.normal(scale=0.3)\n"
    "    w[t] = 0.9 * np.tanh(2 * x[t - 2]) + 0.1 * w[t - 1] + rng.normal(scale=0.3)\n"
    "u = rng.normal(size=n)\n\n"
    "# A random walk, to show the stationarity step doing real work.\n"
    "level = np.cumsum(rng.normal(size=n))\n"
    "df = pd.DataFrame({'x': x, 'y': y, 'w': w, 'u': u, 'level': level})\n"
    "df.head()"
))

c.append(nbf.v4.new_markdown_cell(
    "## 1. Check and enforce stationarity\n\n"
    "Granger tests assume weakly stationary inputs. `adf_report` runs the\n"
    "Augmented Dickey-Fuller test on every column; `make_stationary`\n"
    "differences until each series passes."
))

c.append(nbf.v4.new_code_cell(
    "from grangercm import adf_report, make_stationary\n\n"
    "print('Before:')\n"
    "print(adf_report(df))\n\n"
    "stat_df, orders = make_stationary(df, method='diff')\n"
    "print('\\nDifferencing orders applied:', orders)\n"
    "print('\\nAfter:')\n"
    "print(adf_report(stat_df))"
))

c.append(nbf.v4.new_markdown_cell(
    "Only `level` needed differencing; the others were already stationary.\n\n"
    "## 2. Build the linear Granger Causality Matrix\n\n"
    "Entry `[effect, cause]` is the p-value for *cause Granger-causes effect*.\n"
    "We test lags 1-4, aggregate per pair by the minimum p-value, and apply a\n"
    "Benjamini-Hochberg correction across all off-diagonal cells."
))

c.append(nbf.v4.new_code_cell(
    "from grangercm import granger_matrix\n\n"
    "result = granger_matrix(\n"
    "    stat_df,\n"
    "    variables=['x', 'y', 'w', 'u'],\n"
    "    maxlag=4,\n"
    "    test='ssr_ftest',\n"
    "    aggregate='min',\n"
    "    correction='fdr_bh',\n"
    ")\n"
    "result.pvalues.round(4)"
))

c.append(nbf.v4.new_markdown_cell(
    "The `x` column shows small p-values in the `y` and `w` rows -- exactly\n"
    "the planted links. Note the linear test detects `x -> w` only weakly,\n"
    "because that channel is nonlinear; we revisit it in section 5.\n\n"
    "## 3. Significant edges"
))

c.append(nbf.v4.new_code_cell(
    "result.edges(alpha=0.05)"
))

c.append(nbf.v4.new_markdown_cell(
    "## 4. Visualise as a heatmap\n\n"
    "The heatmap reads like a correlation table. Green = low p-value\n"
    "(significant causality), centered on `alpha`."
))

c.append(nbf.v4.new_code_cell(
    "ax = result.plot(alpha=0.05)\n"
    "plt.savefig('docs/heatmap.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('saved docs/heatmap.png')"
))

c.append(nbf.v4.new_markdown_cell(
    "## 5. Nonlinear backend\n\n"
    "`x -> w` is a `tanh` channel that the linear test underrates. The\n"
    "nonlinear matrix swaps the autoregression for an arbitrary estimator and\n"
    "assesses significance with a block-permutation test. Here we use a small\n"
    "random forest and a reduced permutation count for speed."
))

c.append(nbf.v4.new_code_cell(
    "from grangercm import nonlinear_granger_matrix\n"
    "from sklearn.ensemble import RandomForestRegressor\n\n"
    "nl = nonlinear_granger_matrix(\n"
    "    stat_df,\n"
    "    variables=['x', 'w', 'u'],\n"
    "    maxlag=[2],\n"
    "    estimator=RandomForestRegressor(n_estimators=60, random_state=0),\n"
    "    n_permutations=99,\n"
    "    n_folds=4,\n"
    ")\n"
    "nl.pvalues.round(4)"
))

c.append(nbf.v4.new_markdown_cell(
    "The nonlinear test confirms `x -> w` at lag 2 and leaves the noise\n"
    "variable `u` unflagged in both directions.\n\n"
    "## Caveats\n\n"
    "- Granger causality is *predictive*, not structural, causality.\n"
    "- `aggregate='min'` is conventional but anti-conservative; always pair it\n"
    "  with a `correction`.\n"
    "- The nonlinear backend is much more expensive; restrict `variables`,\n"
    "  `maxlag`, or `n_permutations` for wide panels."
))

nb["cells"] = c

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})

with open("docs/getting_started.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote docs/getting_started.ipynb")
