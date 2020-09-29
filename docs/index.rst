Welcome to active-mri-acquisition's documentation!
==================================================
``active-mri-acquisition`` is a package that facilitates the application of reinforcement learning
to the problem active MRI acquisition. In particular, ``active-mri-acquisition`` provides a gym-like
environment for simulating the execution of policies for k-space sampling, allowing users to
experiment with their own reconstruction models and RL algorithms, without worrying about
implementing the core k-space acquisition logic.

Getting started
===============

Installation
------------

``active-mri-acquisition`` is a Python 3.7+ library. Also, make sure your Python environment has
`PyTorch <https://pytorch.org/>`_ installed with the appropriate CUDA configuration for your system.

To install ``active-mri-acquisition``, clone the repository, then run

.. code-block:: bash

    $ pip install pyxb==1.2.6
    $ pip install -e .

If you also want the developer tools for contributing, run

.. code-block:: bash

    $ pip install -e ".[dev]"

Configuring the environment
---------------------------
To run the environments, you need to configure a couple of things. If you try to run any of the
default environments for the first time (for example, see our `intro notebook
<https://github.com/facebookresearch/active-mri-acquisition/blob/master/notebooks/miccai_example.ipynb>`_),
you will see a message asking you to add some entries to the `defaults.json` file. This file will
be created automatically the first time you run it, located at `$USER_HOME/.activemri/defaults.json`.
It will look like this:

.. code-block:: json

    {
      "data_location": "",
      "saved_models_dir": ""
    }

To run the environments, you need to fill these two entries. Entry `"data_location"` must point to
the root folder in which you will store the fastMRI dataset (for instructions on how to download
the dataset, please visit https://fastmri.med.nyu.edu/). Entry `"saved_models_dir"` indicates the
folder where the environment will look for the checkpoints of reconstruction models.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   envs_intro.rst
   notebooks/miccai_example.ipynb
   custom_reconstructor.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
