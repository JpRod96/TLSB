from plotter import Plotter
from txtGenerator import TxtGenerator

#path = "C:/Users/Jp/Desktop/brazo/bg"
path = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/MLP/results/nuevasSeñas/sigmoidal/4/20aug/testResults.txt"

switcher = {
        1: "Sigmoidal"
    }

plotter = Plotter(switcher)
plotter.plot_txts_from_path(path)


#gen = TxtGenerator(path)
#gen.indices()
