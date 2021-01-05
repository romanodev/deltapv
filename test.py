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
            0.0, 0.02385882199287483, 0.04771764398574966,
            0.07157646597862448, 0.09543528797149932, 0.11929410996437415,
            0.14315293195724896, 0.1670117539501238, 0.19087057594299864,
            0.21472939793587345, 0.2385882199287483, 0.2624470419216231,
            0.2863058639144979, 0.3101646859073728, 0.3340235079002476,
            0.3578823298931224, 0.3817411518859973, 0.4055999738788721,
            0.4294587958717469, 0.45331761786462177, 0.4771764398574966,
            0.5010352618503714, 0.5248940838432462, 0.548752905836121,
            0.5726117278289958, 0.5964705498218708, 0.6203293718147456
        ]

        j_correct = [
            1.0891318612345933e-05, 1.0888610715341167e-05,
            1.088580912438831e-05, 1.0882883580217925e-05,
            1.0879818933660033e-05, 1.087658490761996e-05,
            1.0873166337987491e-05, 1.086947242887586e-05,
            1.0865412351490823e-05, 1.0860804476026559e-05,
            1.0855346071775116e-05, 1.0848598121856516e-05,
            1.0839788544166765e-05, 1.0827690973914394e-05,
            1.0810412672484472e-05, 1.0784803917367823e-05,
            1.0745806526520907e-05, 1.0685181340454917e-05,
            1.0589568856673309e-05, 1.0437246657906412e-05,
            1.0192689397197096e-05, 9.797885034482912e-06,
            9.157845726685026e-06, 8.116401904908686e-06,
            6.41632033331634e-06, 3.631692660287569e-06,
            -9.459292594984389e-07]

        self.assertTrue(np.allclose(v, v_correct), "Voltages do not match!")
        self.assertTrue(np.allclose(j, j_correct), "Currents do not match!")


if __name__ == '__main__':
    unittest.main()
