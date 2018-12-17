import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

with PdfPages('diagnostics/ranked_spectra.pdf') as pdf:
    # Open the complete catalogue
    catalogue = np.load("data/matched_catalogue.npy")

    # Select out cubes with redshift between 0.3 and 0.6
    redshift_cut = catalogue[catalogue[:,7]<=0.6, :]
    redshift_cut = redshift_cut[redshift_cut[:,7]>=0.3, :]

    # Rank by V-band magnitude
    v_band = redshift_cut[redshift_cut[:,5].argsort()]
    
    for i in range(len(v_band)):
        cube_id = int(v_band[i][0])
        v_mag = v_band[i][5]
        
        # I need to open the spectra and save it as a PDF
        x_data_loc = ("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_cbd_x.npy")

        if not os.path.exists(x_data_loc):
            pass
        else:
            x_data = np.load(x_data_loc)
            y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
                    str(int(cube_id)) + "_cbs_y.npy")
    
            plt.figure()
            plt.plot(x_data, y_data, lw=0.5, c="#000000")
            plt.title('Cube ' + str(cube_id) + '| V-band: ' + str(v_mag), fontsize=13)
            plt.tick_params(labelsize=13)
            plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
            plt.ylabel(r'\textbf{Flux}', fontsize=13)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    d = pdf.infodict()
    d['Title'] = 'Ranked Spectra by V-band magnitude'
    d['Author'] = u'Jacky Cao'
    #d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    #d['Keywords'] = 'PdfPages multipage keywords author title subject'
    #d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['CreationDate'] = datetime.datetime.today()

