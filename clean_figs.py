"""Tiny helper script to delete the figures that not show in markdown."""
import os

print("Start to clean the 'figs' directory.")

figs = [(fig.split('.')[0], os.path.join("figs", fig)) for fig in
        os.listdir("figs")]

with open("README.md", 'r') as f:
    content = f.read()

for fig_name, fig_path in figs:
    assert os.path.exists(fig_path)
    if fig_name not in content:
        print("We found that the figure: {} not in content. "
              "So delete the file: {}.".format(fig_name, fig_path))
        os.remove(fig_path)
        assert not os.path.exists(fig_path)

print("Finish to clean the 'figs' directory.")
