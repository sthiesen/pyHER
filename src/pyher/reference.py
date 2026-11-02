"""Reference class weights of the Jura lead example.

``JURA_PB_THRESHOLD_WEIGHTS`` are the optimum OR / AND weights produced by MATLAB ``fmincon`` for
``sHER_E_04_0_Jura_struct.m`` (MATLAB R2018b, threshold objective z_c = log10(50), lag 0.07 km,
bin width 0.015, 21 classes). The OR weights agree with Fig. 14b of Thiesen & Ehret (2021); the AND
weights follow the same step pattern but are slightly lower in the first classes (0.383 vs ~0.41)
because the AND objective is nearly flat. The paper reports alpha = 0.65 and beta = 0.

They are provided so that the paper's model can be reproduced without re-running the optimisation,
see ``optimize_weights(..., fixed_weights=...)``.
"""

import numpy as np

JURA_PB_THRESHOLD_WEIGHTS = {
    "w_or": np.array([
        0.28012326463545434, 0.2579569865544683, 0.10114726045673741,
        0.06797546244739767, 0.04862429486754474, 0.03724065432535681,
        0.0322444648361175, 0.027690299471889716, 0.024184019125474857,
        0.02159448817314194, 0.019000598353717747, 0.01644275734703892,
        0.014686603431235118, 0.012963583446467176, 0.010813227670942019,
        0.008675409705295853, 0.007047220814207864, 0.005427653124850196,
        0.0038597121662618007, 0.002173096012366154, 0.00012894303403386838,
    ]),
    "w_and": np.array([
        0.38301917371777877, 0.38298383081161924, 0.18734483670260582,
        0.18733472266142834, 0.18719495468344793, 0.025929044493089515,
        0.025925195249221024, 0.025896514366372806, 0.025598621597233487,
        0.025593145129184775, 0.025581657854674162, 0.025576297352587526,
        0.025574108755371937, 0.02557115421468284, 0.01612407364887679,
        0.01611424821724725, 0.016106744865096, 0.016090990044017914,
        0.01608230088972534, 0.006073069093760873, 0.006068288315560718,
    ]),
}
JURA_PB_PAPER_ALPHA_BETA = (0.65, 0.0)
