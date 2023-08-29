# nlpsig-networks

<p align="center">
<img src="./fig/architecture_figure.png" alt="Model architecture">
</p>

Work in progress repo.

First create an environment and install
[`nlpsig`](https://github.com/datasig-ac-uk/nlpsig), and then install this
package afterwards:

```
git clone git@github.com:datasig-ac-uk/nlpsig.git
git clone git@github.com:ttseriotou/nlpsig-networks.git
cd nlpsig
conda env create --name nlpsig-networks python=3.8
conda activate nlpsig-networks
pip install -e .
cd ../nlpsig-networks
pip install -e .
```

If you want to install a development version, use `pip install -e .` instead in
the above.

## Pre-commit and linters

To take advantage of `pre-commit`, which will automatically format your code and run some basic checks before you commit:

```
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format the codebase. You can also/alternatively run `pre-commit run --all-files` to run the checks.

