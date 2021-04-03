import unittest
import deltapv as dpv
from jax import numpy as jnp

L = 3e-4
J = 5e-6
material = dpv.create_material(Chi=3.9,
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
design = dpv.make_design(n_points=500,
                         Ls=[J, L - J],
                         mats=[material, material],
                         Ns=[1e17, -1e15],
                         Snl=1e7,
                         Snr=0,
                         Spl=0,
                         Spr=1e7)
ls = dpv.incident_light()


class TestDeltaPV(unittest.TestCase):
    def test_iv(self):

        results = dpv.simulate(design, ls)
        v, j = results["iv"]

        v_correct = [
            0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25,
            0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55,
            0.6000000000000001, 0.6500000000000001, 0.7000000000000001, 0.75,
            0.8, 0.85, 0.9, 0.9500000000000001
        ]

        j_correct = [
            0.01882799450659129, 0.018753370994746384, 0.018675073222852775,
            0.018592788678882418, 0.01850616015841796, 0.018414776404918568,
            0.018318159501526814, 0.01821574845029824, 0.018106874755825324,
            0.0179907188741479, 0.017866203205496447, 0.017731661626627034,
            0.017583825887487907, 0.01741498506998538, 0.017204823904941775,
            0.01689387681804267, 0.01628556057166174, 0.014630769395991339,
            0.008610345709349041, -0.018267911703588706
        ]

        self.assertTrue(jnp.allclose(v, v_correct), "Voltages do not match!")
        self.assertTrue(jnp.allclose(j, j_correct), "Currents do not match!")


if __name__ == '__main__':
    unittest.main()
