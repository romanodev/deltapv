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

        v, j = jaxpv.simulator.iv_curve(design, ls)

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
            1.0542663641677582e-05, 1.054004432002233e-05,
            1.053734651862992e-05, 1.0534494623040321e-05,
            1.0531549081188675e-05, 1.0528449383290128e-05,
            1.0525119873577099e-05, 1.0521560495250588e-05,
            1.051760479841357e-05, 1.051311652894042e-05,
            1.0507792960063945e-05, 1.050116481723714e-05,
            1.0492490278375705e-05, 1.0480542982591286e-05,
            1.046341506026371e-05, 1.0437971924921247e-05,
            1.039912512316864e-05, 1.0338696047065337e-05,
            1.0243279785131078e-05, 1.0091123636554666e-05,
            9.846762818062418e-06, 9.452185297413967e-06, 8.81235780230008e-06,
            7.771186494602175e-06, 6.071347264192148e-06,
            3.287022649198452e-06, -1.2903036594973483e-06
        ]

        self.assertTrue(np.allclose(v, v_correct), "Voltages do not match!")
        self.assertTrue(np.allclose(j, j_correct), "Currents do not match!")


if __name__ == '__main__':
    unittest.main()
