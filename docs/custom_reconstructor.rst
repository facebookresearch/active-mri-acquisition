Adding your own reconstructor model
===================================

Configuration file
------------------


Reconstructor interface
-----------------------
A reconstructor model is essentially a ``torch.nn.Module`` with one additional method,
``init_from_checkpoint``, which receives a model checkpoint as dictionary and initializes the
model accordingly.

.. literalinclude:: ../activemri/models/__init__.py
    :lines: 12-21

All of our ``active-mri-acquisition`` environments will perform the following steps:

1. Instantiate reconstructor class.
2. Load the checkpoint as ``checkpoint_dict = torch.load(checkpoint_path)``.
3. Call ``reconstructor.init_from_checkpoint(checkpoint_dict)``.

The other important convention we adopt is that ``forward()`` will return a
dictionary with the output of the model. This dictionary **must** contain key ``"reconstruction"``,
whose value is the reconstructed image. However, the model can also return other outputs that
will also be returned as part of the observation, as explained in :ref