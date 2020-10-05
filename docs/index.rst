Documentation for active-mri-acquisition
========================================
``active-mri-acquisition`` is a package that facilitates the application of reinforcement learning
to the problem active MRI acquisition. In particular, ``active-mri-acquisition`` provides a gym-like
environment for simulating the execution of policies for k-space sampling, allowing users to
experiment with their own reconstruction models and RL algorithms, without worrying about
implementing the core k-space acquisition logic.

Getting started
===============

Installation
------------

``active-mri-acquisition`` is a Python 3.7+ library. To install it, clone the repository,

.. code-block:: bash

    git clone https://github.com/facebookresearch/active-mri-acquisition.git

then run

.. code-block:: bash

    cd active-mri-acquisition
    pip install -e .

If you also want the developer tools for contributing, run

.. code-block:: bash

    pip install -e ".[dev]"

Finally, make sure your Python environment has
`PyTorch (>= 1.6) <https://pytorch.org/>`_ installed with the appropriate CUDA configuration
for your system.


To test your installation, run

.. code-block:: bash

    python -m pytest tests/core

.. _configuring-activemri:

Global configuration
--------------------
The first time you try to run any of our RL environments (for example, see our `intro notebook
<https://github.com/facebookresearch/active-mri-acquisition/blob/master/notebooks/miccai_example.ipynb>`_),
you will see a message asking you to add some entries to the `defaults.json` file. This file will
be created automatically the first time you run an environemnt, and it will be located at
`$HOME/.activemri/defaults.json`. It will look like this:

.. code-block:: json

    {
      "data_location": "",
      "saved_models_dir": ""
    }

To run the RL environments, you need to fill these two entries. Entry ``data_location`` must point
to the root folder in which you will store the fastMRI dataset (for instructions on how to download
the dataset, please visit https://fastmri.med.nyu.edu/). Entry ``saved_models_dir`` indicates the
folder where the environment will look for the checkpoints of reconstruction models.  Note that
``saved_models_dir`` does not need to be set to use your own reconstruction model, but it is
required to use our example environments. For more details see :ref:`JSON_config`.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   ../notebooks/miccai_example.ipynb
   create_env.rst
   custom_reconstructor.rst
   api.rst
   misc.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
