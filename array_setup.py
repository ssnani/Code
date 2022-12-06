from collections import namedtuple
import numpy as np

# Named tuple with the characteristics of a microphone array and definitions of the LOCATA arrays:
ArraySetup = namedtuple('ArraySetup', 'arrayType, orV, mic_pos, mic_orV, mic_pattern')

array_setup_10cm_2mic = ArraySetup(arrayType='planar', 
                                    orV = np.array([0.0, 0.0, 1.0]),
                                    mic_pos = np.array((( 0.05,  0.000, 0.000),
                                                        (-0.05,  0.000, 0.000))), 
                                    mic_orV = np.array(((0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0))), 
                                    mic_pattern = 'omni'
                                )