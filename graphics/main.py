from plotter import Plotter
from txtGenerator import TxtGenerator

path = "C:/Users/Jp/Desktop/brazo/bg"
"""
plotter = Plotter()
plotter.plot_txts_from_path(path)
"""

gen = TxtGenerator(path)
gen.indices()
