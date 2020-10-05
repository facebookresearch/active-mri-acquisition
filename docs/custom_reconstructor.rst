Adding your own reconstructor model
===================================

Configuring the RL environment to use your own reconstruction model involves three steps:

    1. Create a transform function to convert data loader outputs into an batched input for your
       reconstructor model.
    2. Refactor your reconstructor model to follow our Reconstructor interface.
    3. Modify the environment's JSON configuration file accordingly.

We explain the first two steps below. The third step is explained in :ref:`JSON_config`.

.. _transform_fn:

Transform function
------------------
Communication between the fastMRI data loader and your reconstructor is done via a
transform function, which should follow the signature of :meth:`~activemri.data.transform_template`.
The environment will first load data from the fastMRI dataset, and collate it to meet the input
format indicated in the documentation of :meth:`~activemri.data.transform_template`.
The environment will then pass this input to your provided transform function (as separate
keyword arguments), and subsequently pass the output of the transform to the reconstructor model.
The complete sequence will roughly conform to the following pseudocode.

.. code-block:: python

    kspace, _, ground_truth, attrs, fname, slice_id = data_handler[item]
    mask = get_current_active_mask()
    reconstructor_input = transform(
        kspace=kspace,
        mask=mask,
        ground_truth=ground_truth,
        attrs=attrs,
        fname=fname,
        slice_id=slice_id
    )
    reconstructor_output = reconstructor(*reconstructor_input)

Some examples of transform functions are available:

    * `Transform <https://github.com/facebookresearch/active-mri-acquisition/blob/master/activemri/data/transforms.py#L66>`_
      used in our `MICCAI 2020 paper <https://arxiv.org/pdf/2007.10469.pdf>`_.
    * `Example transform <https://github.com/facebookresearch/active-mri-acquisition/blob/master/activemri/data/transforms.py#L164>`_
      for using fastMRI's `Unet model <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py>`_
      on single coil data. This example shows how to handle k-space of variable width (in the
      case where the environment is created with multiple ``num_cols`` values).

.. note::
    If your reconstructor only needs as input a zero-filled reconstruction (i.e, inverse Fourier
    transform for non-zero k-space columns), and perhaps mean and standard deviation to use
    for normalization, then
    :func:`~activemri.data.transforms.fastmri_unet_transform_singlecoil` and
    :func:`~activemri.data.transforms.fastmri_unet_transform_multicoil` should be a good place
    to start for your own transform.

Reconstructor interface
-----------------------
A reconstructor model is essentially a ``torch.nn.Module`` that must follow a few additional
conventions. In terms of the class interface, besides the usual ``torch.nn.Module`` method,
the reconstructor must also include a method ``init_from_checkpoint``, which receives a model
checkpoint as dictionary and initializes the model from this data.

.. literalinclude:: ../activemri/models/__init__.py
    :lines: 12-21

The other conventions concern the model intialization and the output format of the forward method.
We explain these below.

Initializing reconstructor
^^^^^^^^^^^^^^^^^^^^^^^^^^
The environment expects the reconstructor to be passed  initialization arguments as keywords.
The set of keywords and their values will be read from the environment's configuration file.
However if your checkpoint dictionary contains a key called ``"options"``, then this will take
precedence, and the environment will emit a warning.
The sequence will roughly conform to the following pseudocode:

.. code-block:: python

    reconstructor_cls, reconstructor_cfg_dict, checkpoint_path = read_from_env_config()
    checkpoint_dict = torch.load(checkpoint_path)
    reconstructor_cfg = override_if_options_key_present(checkpoint_dict)
    reconstructor = reconstructor_cls(**reconstructor_cfg_dict)
    reconstructor.init_from_checkpoint(checkpoint_dict)  # load weights, additional bookkeeping

Forward signature
^^^^^^^^^^^^^^^^^

The other important convention we adopt is that ``forward()`` will return a
dictionary with the output of the model. This dictionary **must** contain key ``"reconstruction"``,
whose value is the reconstructed image tensor. Note that the model can also return additional
outputs, which will also included in the observation returned by the environment, as explained in
`the basic example <notebooks/miccai_example.ipynb>`_.

Examples
^^^^^^^^

Some examples are available at the
`models directory <https://github.com/facebookresearch/active-mri-acquisition/tree/new_master/activemri/models>`_,
in the repository. Note that no changes to the reconstructor model are required, and the coupling
between environment and reconstructor can be done via short wrapper classes.
