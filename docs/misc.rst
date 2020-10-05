Miscellanea
===========

Available checkpoints
---------------------
Use the commands below to download the model checkpoints used to produce the results in
Pineda et al., MICCAI'20.

Reconstructor checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^

For normal acceleration ("Scenario30L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_reconstructor_raw_normal.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_reconstructor_raw_normal.md5
    md5sum -c miccai2020_reconstructor_raw_normal.md5

For extreme acceleration ("Scenario2L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_reconstructor_raw_extreme.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_reconstructor_raw_extreme.md5
    md5sum -c miccai2020_reconstructor_raw_extreme.md5

Evaluator checkpoints
^^^^^^^^^^^^^^^^^^^^^

For normal acceleration ("Scenario30L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_evaluator_raw_normal.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_evaluator_raw_normal.md5
    md5sum -c miccai2020_evaluator_raw_normal.md5

For extreme acceleration ("Scenario2L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_evaluator_raw_extreme.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_evaluator_raw_extreme.md5
    md5sum -c miccai2020_evaluator_raw_extreme.md5

SS-DDQN checkpoints (subject-specific variant)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For normal acceleration ("Scenario30L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_ss-ddqn_ssim_normal.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_ss-ddqn_ssim_normal.md5
    md5sum -c miccai2020_ss-ddqn_ssim_raw_normal.md5

For extreme acceleration ("Scenario2L" in the paper)

.. code-block:: bash

    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_ss-ddqn_ssim_extreme.pth
    # To verify
    wget https://dl.fbaipublicfiles.com/active-mri-acquisition/miccai2020_ss-ddqn_ssim_extreme.md5
    md5sum -c miccai2020_ss-ddqn_ssim_raw_extreme.md5
