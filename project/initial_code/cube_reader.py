import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

from astropy.io import fits

def read_file(file_name):
    # reads file_name and returns specific header data and image data

    fits_file = fits.open(file_name)

    header = fits_file[0].header
    image_data = fits_file[0].data

    header_keywords = {'CRVAL3': 0, 'CRPIX3': 0, 'CD3_3': 0}
    # clause to differentiate between CDELT3 and CD3_3

    for hdr_key, hdr_value in header_keywords.items():
        # finding required header values
        hdr_value = header[hdr_key]
        header_keywords[hdr_key] = hdr_value

    return header_keywords, image_data


def wavelength_solution(file_name):
    # wavelength solution in Angstroms

    file_data   = read_file(file_name)
    header_data = file_data[0]
    image_data  = file_data[1]

    range_begin = header_data['CRVAL3']
    pixel_begin = header_data['CRPIX3']
    step_size   = header_data['CD3_3']

    range_end   = range_begin + len(image_data) * step_size

    return range_begin, range_end

def image_collapser(file_name):

    file_data   = read_file(file_name)
    header_data = file_data[0]
    image_data  = file_data[1]

    data_shape  = np.shape(image_data)
    ra_axis     = data_shape[2]
    dec_axis    = data_shape[1]
    wl_axis     = data_shape[0]
    
    image_median = np.zeros((ra_axis, dec_axis))
    image_sum = np.zeros((ra_axis, dec_axis))

    for i_ra in range(ra_axis):
        for i_dec in range(dec_axis):
            pixel_data  = image_data[:][:,i_dec][:,i_ra]
            pd_median   = np.median(pixel_data)
            pd_sum      = np.sum(pixel_data)

            image_median[i_ra][i_dec]   = pd_median
            image_sum[i_ra][i_dec]      = pd_sum

    plt.imshow(image_median, cmap='gray')
    plt.colorbar()
    plt.show()


#initial = image_collapser("cube_23.fits")
#print(initial[1])

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QVBoxLayout()
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self):
        file_name   = "cube_23.fits"
        file_data   = read_file(file_name)
        header_data = file_data[0]
        image_data  = file_data[1]

        data_shape  = np.shape(image_data)
        ra_axis     = data_shape[2]
        dec_axis    = data_shape[1]
        wl_axis     = data_shape[0]
        
        image_median = np.zeros((ra_axis, dec_axis))
        image_sum = np.zeros((ra_axis, dec_axis))

        for i_ra in range(ra_axis):
            for i_dec in range(dec_axis):
                pixel_data  = image_data[:][:,i_dec][:,i_ra]
                pd_median   = np.median(pixel_data)
                pd_sum      = np.sum(pixel_data)

                image_median[i_ra][i_dec]   = pd_median
                image_sum[i_ra][i_dec]      = pd_sum

        self.figure.clear()

        #image_median = np.flipud(image_median)

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        plt.imshow(image_median, cmap='gray')
        plt.colorbar()

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
