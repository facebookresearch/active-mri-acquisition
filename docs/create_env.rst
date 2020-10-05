Configuring the environment
===========================

Instantiating an environment
----------------------------
As an entry point for active acquisition with fastMRI data, we provide class
:class:`~activemri.envs.envs.FastMRIEnv`. The information about what reconstructor model to use,
and how to use it, is passed via a small JSON configuration file.
The following example illustrates how to use this class to create an RL environment for the
fastMRI singlecoil knee data:

.. code-block:: python

    from activemri.envs import FastMRIEnv

    env = FastMRIEnv(
        "path/to/cfg.json",
        "knee_singlecoil",
        num_parallel_episodes=2,
        budget=70,
        num_cols=(368, 372)
    )

Here, the environment reads its configuration from ``""path/to/cfg.json""``. The remaining
arguments indicate that the fastMRI dataset to use is ``"knee_singlecoil"``, very episode will
operate on a batch of 2 images (``num_parallel_episodes``), episodes will last for 70 acquisition
steps (``budget=70``), and the fastMRI dataset will return images with k-space having both 368
and 372 columns.

Within the dataset folder, the environment will look for directories suffixed by ``"_train"``,
``"_val"`` and ``"_test"`` (e.g., ``"knee_singlecoil_train"``, ``"multicoil_val"``, and so on).
The first two are mandatory, but if there is no test split, the environment will default to use
validation split for testing and emit a warning.

.. _JSON_config:

Environment's JSON configuration
--------------------------------
The RL environment uses a small JSON file to figure out the following configuration. The template
for all configuration files is the following:

.. code-block:: json

    {
        "data_location": "",
        "reconstructor": {
            "cls": "",
            "options": {
            },
            "checkpoint_fname": "",
            "transform": ""
        },
        "mask": {
            "function": "",
                "args": {
                }
        },
        "reward_metric": "",
        "device": ""
    }

The meaning of the attributes is the following:

    * ``"data_location"``: A string describing the location of the fastMRI dataset root folder.
      The environment will look for each specific dataset under this location
      (e.g., `data_location/knee_singlecoil_train`, `data_location/brain_multicoil_train`). If the
      provided value is not a valid directory, the environment will use the default config
      (see :ref:`configuring-activemri`).
    * ``"reconstructor"``: Reconstructor configuration.
      It's a dictionary with the following entries:

        * ``"cls"``: The class name of the reconstructor (e.g., ``parent.module.Reconstructor``).
        * ``"options"``: A dictionary with the arguments and values to be passed to the
          reconstructor's ``__init__()`` method, as keyword arguments.
        * ``"checkpoint_fname"``: The name of the file that stores the reconstructor's checkpoint
          (e.g., to load weights). As explained in :ref:`configuring-activemri`, the environment
          code will look for the model under folder ``saved_model_dir``, specifically in path
          ``saved_models_dir/checkpoint_fname``. However, if ``checkpoint_fname`` is an absolute
          path, the environment will ignore the path in ``saved_models_dir`` and just use
          ``checkpoint_fname``.
        * ``"transform"``: The function name of the transform to use to convert fastMRI data into
          and input to the reconstructor model. For details see :ref:`transform_fn`.

    * ``"mask"``: Configuration for the initial masks, indicating the active k-space
      columns at the beginning of the episodes. It should be a dictionary with the following
      entries:

        * ``"function"``: The name of the mask function (e.g., ``parent.module.my_mask_fn``).
        * ``"args"``: A dictionary with configurations options for the mask function.

      To see available masks, please see :ref:`our API<mask_api>`. You can also use your
      custom initial masks if needed.
    * ``"reward_metric"``: Which error metric the environment will use as a reward. Valid option
      are ``"mse", "ssim", "nmse", "psnr"``.
    * ``"device"``: ``torch`` device to use for the reconstructor model.

We provide some sample configuration files under the repository's
`configs folder <https://github.com/facebookresearch/active-mri-acquisition/tree/master/configs>`_.

.. warning::
    You need to make sure that ``reconstructor_cls`` and ``mask.function`` (if using a custom mask)
    are importable, for example by installing them in your virtualenv, or by adding them to
    ``PYTHONPATH`` before calling the environment. For convenience, our ``git`` setup ignores
    files named as ``activemri/models/custom_*`` and ``activemri/masks/custom_*``, so an easy
    option, available if you have installed using ``pip install -e .``, is to add them to these
    folders using the above naming convention. Then they can be addressed as, for example,
    ``activemri.modules.custom_reconstructor.MyReconstructor``.

Pre-configured environments
---------------------------

As examples, we provide some few pre-configured subclasses of :class:`~activemri.envs.envs.FastMRIEnv`:

    * :class:`~activemri.envs.envs.SingleCoilKneeEnv`: Uses the fastMRI ``"knee_singlecoil"``
      dataset and fastMRI's `Unet model <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py>`_.
      The config file is `here <https://github.com/facebookresearch/active-mri-acquisition/tree/master/configs/single-coil-knee.json>`_.

    * :class:`~activemri.envs.envs.MultiCoilKneeEnv`: Uses the fastMRI ``"multicoil"``
      dataset and fastMRI's `Unet model <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py>`_.
      The config file is `here <https://github.com/facebookresearch/active-mri-acquisition/tree/master/configs/multi-coil-knee.json>`_.

These environments are configured to start episodes with a constant number of active low frequencies
(15 active columns on each side), and will use images with k-spaces of 368 and 372 lines. Check the
documentation of low frequency mask function to see how variable k-spaces are handled. This can also
be turned off by creating the environments filtering for a single value, e.g.,
``num_cols=(368,)``, and setting ``mask["args]["max_width"]`` to the chosen value, in the JSON config.