from dataSetCharger import DataSetCharger

dsc = DataSetCharger()

path = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/CBBA/20190812_160520EdgesAugmented1.jpg"
print(dsc.image_to_lbp(path, 24, 8))

path = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/CBBA/VID_20190811_130522EdgesAugmented5.jpg"
print(dsc.image_to_lbp(path, 24, 8))
