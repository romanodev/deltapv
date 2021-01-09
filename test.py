import unittest
import jaxpv
from jax import numpy as np

L = 3e-4
grid = np.linspace(0, L, 500)
design = jaxpv.simulator.create_design(grid)
material = jaxpv.materials.create_material(Chi=3.9,
                                           Eg=1.5,
                                           eps=9.4,
                                           Nc=8e17,
                                           Nv=1.8e19,
                                           mn=100,
                                           mp=100,
                                           Et=0,
                                           tn=1e-8,
                                           tp=1e-8,
                                           A=1e4)
design = jaxpv.simulator.add_material(design, material, lambda x: True)
design = jaxpv.simulator.contacts(design, 1e7, 0, 0, 1e7)
design = jaxpv.simulator.single_pn_junction(design, 1e17, -1e15, 50e-7)
ls = jaxpv.simulator.incident_light()


class TestJAXPV(unittest.TestCase):
    def test_iv(self):

        results = jaxpv.simulator.simulate(design, ls)
        v, j = results["iv"]

        v_correct = [
            0.0, 0.02385882199287483, 0.04771764398574966, 0.07157646597862448,
            0.09543528797149932, 0.11929410996437415, 0.14315293195724896,
            0.1670117539501238, 0.19087057594299864, 0.21472939793587345,
            0.2385882199287483, 0.2624470419216231, 0.2863058639144979,
            0.3101646859073728, 0.3340235079002476, 0.3578823298931224,
            0.3817411518859973, 0.4055999738788721, 0.4294587958717469,
            0.45331761786462177, 0.4771764398574966, 0.5010352618503714,
            0.5248940838432462, 0.548752905836121, 0.5726117278289958,
            0.5964705498218708, 0.6203293718147456
        ]

        j_correct = [
            9.540488439618553e-06, 9.537609815434845e-06, 9.53462265523875e-06,
            9.531511814964217e-06, 9.52827726623705e-06, 9.524888735798109e-06,
            9.521285691632699e-06, 9.517422705066578e-06,
            9.513193824438541e-06, 9.508387163496693e-06,
            9.502745371206822e-06, 9.495799172882766e-06,
            9.486806746299358e-06, 9.4745417304272e-06, 9.457065985724803e-06,
            9.431275189921562e-06, 9.39209602710166e-06, 9.331304476387006e-06,
            9.23554103898328e-06, 9.083037876349409e-06, 8.83833020508617e-06,
            8.443390848494303e-06, 7.803231967148943e-06,
            6.7617142898103315e-06, 5.061543998916609e-06,
            2.2768733371612674e-06, -2.3007534046630316e-06
        ]

        self.assertTrue(np.allclose(v, v_correct), "Voltages do not match!")
        self.assertTrue(np.allclose(j, j_correct), "Currents do not match!")


if __name__ == '__main__':
    unittest.main()
