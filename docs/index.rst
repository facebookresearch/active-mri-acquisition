Welcome to active-mri-acquisition's documentation!
==================================================
``active-mri-acquisition`` is a package that facilitates the application of reinforcement learning
to the problem active MRI acquisition. In particular, ``active-mri-acquisition`` provides a gym-like
environment for simulating the execution of policies for k-space sampling, allowing users to
experiment with their own reconstruction models and RL algorithms, without worrying about
implementing the core k-space acquisition logic.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: activemri.envs.envs
    :members:
    :noindex:

.. automodule:: activemri.envs.masks
    :members:
    :noindex:

.. automodule:: cvpr19_models.models.reconstruction
    :members:
    :noindex:

.. automodule:: cvpr19_models.models.evaluator
    :members:
    :noindex:

.. automodule:: activemri.data.transforms
    :members:
    :noindex:

.. automodule:: activemri.baselines.simple_baselines
    :members:
    :noindex:

.. automodule:: activemri.baselines.ddqn
    :members:
    :noindex:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
