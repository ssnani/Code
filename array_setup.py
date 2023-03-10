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

array_setup_8cm_2mic = ArraySetup(arrayType='planar', 
                                    orV = np.array([0.0, 0.0, 1.0]),
                                    mic_pos = np.array((( -0.04,  0.000, 0.000),
                                                        (  0.04,  0.000, 0.000))), 
                                    mic_orV = np.array(((0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0))), 
                                    mic_pattern = 'omni'
                                )

array_setup_8cm_8mic = ArraySetup(arrayType='planar', 
                                    orV = np.array([0.0, 0.0, 1.0]),
                                    mic_pos = np.array((( -0.28,  0.000, 0.000),
                                                        ( -0.20,  0.000, 0.000),
                                                        ( -0.12,  0.000, 0.000),
                                                        ( -0.04,  0.000, 0.000),
                                                        (  0.04,  0.000, 0.000),
                                                        (  0.12,  0.000, 0.000),
                                                        (  0.20,  0.000, 0.000),
                                                        (  0.28,  0.000, 0.000), )), 
                                    mic_orV = np.array(((0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0))), 
                                    mic_pattern = 'omni'
                                )

array_setup_8cm_4mic = ArraySetup(arrayType='planar', 
                                    orV = np.array([0.0, 0.0, 1.0]),
                                    mic_pos = np.array((( -0.12,  0.000, 0.000),
                                                        ( -0.04,  0.000, 0.000),
                                                        (  0.04,  0.000, 0.000),
                                                        (  0.12,  0.000, 0.000) )), 
                                    mic_orV = np.array(((0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0),
                                                        (0.0, 0.0, 1.0))), 
                                    mic_pattern = 'omni'
                                )

def get_array_set_up_from_config(array_type: str, num_mics: int, intermic_dist: float):
    if 'linear'==array_type:
        if 2==num_mics:
            if 10 ==intermic_dist:
                return array_setup_10cm_2mic
            elif 8 ==intermic_dist:
                return array_setup_8cm_2mic
            else:
                print("Unsuppoted Mic array\n")
        elif 4==num_mics:
            #uniform intermic dist
            if 8 ==intermic_dist:
                return array_setup_8cm_4mic
        elif 8==num_mics:
            #uniform intermic dist
            if 8 ==intermic_dist:
                return array_setup_8cm_8mic
        else:
            print("Unsuppoted Mic array\n")
    else:
        print("Unsuppoted Mic array\n")
    return