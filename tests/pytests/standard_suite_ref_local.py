import numpy as np
from qcengine.programs.tests.standard_suite_ref import answer_hash, compute_derived_qcvars, _std_suite, _std_generics


# in-repo extensions for _std_suite above
# * ideally empty. PR to QCEngine ASAP and empty this after QCEngine release.
_std_suite_psi4_extension = [
    # <<<  CONV-AE-CONV  >>>
    {
        "meta": {
            "system": "hf",
            "basis": "cc-pvdz",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "aug-cc-pvdz",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "cfour-qz2p",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "bh3p",
            "basis": "cc-pvdz",
            "scf_type": "pk",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "aug-cc-pvdz",
            "scf_type": "pk",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "cfour-qz2p",
            "scf_type": "pk",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "bh3p",
            "basis": "cc-pvdz",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.08142433,  # detci != cfour's vcc ???  # locally, replacing the rohf cisd vcc=tce value (stored in qcng) by the detci=guga value. correct sdsc label unclear.
            "FCI CORRELATION ENERGY": -0.084637876308811,  # detci
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "aug-cc-pvdz",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.1723668643052676,  # detci != vcc ???
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "cfour-qz2p",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.21038651,  # detci != vcc ???
        },
    },
    {
        "meta": {
            "system": "hf",
            "basis": "cc-pvdz",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
            "SVWN TOTAL HESSIAN": np.array(
                [
                    -0.011158529195,
                    -0.,
                    -0.,
                    0.011158509954,
                    -0.,
                    -0.,
                    -0.,
                    -0.011158529195,
                    -0.,
                    0.,
                    0.011158509954,
                    0.,
                    -0.,
                    -0.,
                    0.642213454497,
                    0.,
                    0.,
                    -0.642213457165,
                    0.011158509954,
                    0.,
                    0.,
                    -0.011155887562,
                    -0.,
                    -0.,
                    -0.,
                    0.011158509954,
                    0.,
                    -0.,
                    -0.011155887564,
                    -0.,
                    -0.,
                    0.,
                    -0.642213457165,
                    -0.,
                    -0.,
                    0.642216280292,
                ]).reshape((6,6)),
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "aug-cc-pvdz",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
            "SVWN TOTAL HESSIAN": np.array(
                [
                     -0.015282137109,
                     0.,
                     -0.,
                     0.007667256256,
                     0.,
                     -0.,
                     0.007667256256,
                     0.,
                     -0.,
                     0.,
                     0.714169890673,
                     0.000000000001,
                     -0.,
                     -0.357096356198,
                     0.282430177213,
                     -0.,
                     -0.357096356198,
                     -0.282430177213,
                     -0.,
                     0.000000000001,
                     0.45653162932,
                     0.,
                     0.223223468116,
                     -0.228270944088,
                     -0.,
                     -0.223223468116,
                     -0.228270944088,
                     0.007667256256,
                     -0.,
                     0.,
                     -0.008305196866,
                     0.,
                     0.,
                     0.000637932765,
                     -0.,
                     0.,
                     0.,
                     -0.357096356198,
                     0.223223468116,
                     0.,
                     0.385967088688,
                     -0.252826811188,
                     -0.,
                     -0.028870808826,
                     0.029603353817,
                     -0.,
                     0.282430177213,
                     -0.228270944088,
                     0.,
                     -0.252826811188,
                     0.219004239874,
                     0.,
                     -0.029603353817,
                     0.009266633228,
                     0.007667256256,
                     -0.,
                     -0.,
                     0.000637932765,
                     -0.,
                     0.,
                     -0.008305196866,
                     -0.,
                     0.,
                     0.,
                     -0.357096356198,
                     -0.223223468116,
                     -0.,
                     -0.028870808826,
                     -0.029603353817,
                     -0.,
                     0.385967088688,
                     0.252826811188,
                     -0.,
                     -0.282430177213,
                     -0.228270944088,
                     0.,
                     0.029603353817,
                     0.009266633228,
                     0.,
                     0.252826811188,
                     0.219004239874,
                ]).reshape((9, 9)),
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "cfour-qz2p",
            "scf_type": "pk",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "conv",
            "sdsc": "sd"
        },
        "data": {
            "SVWN TOTAL HESSIAN": np.array(
                [
                     -0.010015602973,
                     -0.000000000001,
                     0.000000000002,
                     0.005034918743,
                     0.,
                     -0.,
                     0.005034918743,
                     -0.,
                     0.,
                     -0.000000000001,
                     0.699200093422,
                     -0.000000000002,
                     -0.,
                     -0.349611743856,
                     0.274596771974,
                     -0.,
                     -0.349611743856,
                     -0.274596771974,
                     0.000000000002,
                     -0.000000000002,
                     0.449624843917,
                     -0.,
                     0.216250731357,
                     -0.224817271283,
                     -0.,
                     -0.216250731357,
                     -0.224817271283,
                     0.005034918743,
                     -0.,
                     -0.,
                     -0.005839372129,
                     -0.,
                     0.,
                     0.000803489613,
                     0.,
                     0.,
                     0.,
                     -0.349611743856,
                     0.216250731357,
                     -0.,
                     0.379671911068,
                     -0.245424063175,
                     0.,
                     -0.030060195243,
                     0.029173021276,
                     -0.,
                     0.274596771974,
                     -0.224817271283,
                     0.,
                     -0.245424063175,
                     0.215395274362,
                     0.,
                     -0.029173021276,
                     0.009421810131,
                     0.005034918743,
                     -0.,
                     -0.,
                     0.000803489613,
                     0.,
                     0.,
                     -0.005839372129,
                     0.,
                     -0.,
                     -0.,
                     -0.349611743856,
                     -0.216250731357,
                     0.,
                     -0.030060195243,
                     -0.029173021276,
                     0.,
                     0.379671911068,
                     0.245424063175,
                     0.,
                     -0.274596771974,
                     -0.224817271283,
                     0.,
                     0.029173021276,
                     0.009421810131,
                     -0.,
                     0.245424063175,
                     0.215395274362,
                ]).reshape((9, 9)),
        },
    },

    # <<<  CONV-FC-CONV  >>>
    {
        "meta": {
            "system": "bh3p",
            "basis": "cc-pvdz",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "fc",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.08045048714872,  # detci only != vcc ???
            "FCI CORRELATION ENERGY": -0.083612606639434,  # detci
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "aug-cc-pvdz",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "fc",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.170209639586457,  # detci only != vcc ???
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "cfour-qz2p",
            "scf_type": "pk",
            "reference": "rohf",
            "fcae": "fc",
            "corl_type": "conv",
            "sdsc": "sd",
        },
        "data": {
            "CISD CORRELATION ENERGY": -0.186640254417867,  # detci only != vcc ???
        },
    },

    # <<<  DF-AE-DF  >>>
    {
        "meta": {
            "system": "hf",
            "basis": "cc-pvdz",
            "scf_type": "df",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "h2o",
            "basis": "cfour-qz2p",
            "scf_type": "df",
            "reference": "rhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "bh3p",
            "basis": "cc-pvdz",
            "scf_type": "df",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd",
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    {
        "meta": {
            "system": "nh2",
            "basis": "cfour-qz2p",
            "scf_type": "df",
            "reference": "uhf",
            "fcae": "ae",
            "corl_type": "df",
            "sdsc": "sd"
        },
        "data": {
        },
    },
    # <<<  CD-AE-CD  >>>
    # <<<  CD-FC-CD  >>>
]


for calc1 in _std_suite_psi4_extension:
    metahash1 = answer_hash(**calc1["meta"])
    for calc0 in _std_suite:
        metahash0 = answer_hash(**calc0["meta"])
        if metahash0 == metahash1:
            calc0["data"].update(calc1["data"])
            break

compute_derived_qcvars(_std_suite)
std_suite = {answer_hash(**calc["meta"]): calc["data"] for calc in _std_suite}
