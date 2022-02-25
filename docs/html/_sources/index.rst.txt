.. bvmatrix documentation master file, created by
   sphinx-quickstart on Fri Feb 25 20:11:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bvmatrix's documentation!
====================================

This libary is focused on the iterative solution of the linear system

.. math::

   M\, \mathbf{y} = \mathbf{e}_1 \otimes \eta + h\,(B \otimes I) \mathbf{g},

where

.. math::

    {\bf e}_1=(1,0,\ldots,0)^* \in \mathbb{R}^{s+1}, \quad
   \mathbf{y} = (\mathbf{y}_0,\ldots,\mathbf{y}_s)^*,
   \quad \mathbf{g}=(\mathbf{g}_0,\ldots,\mathbf{g}_s)^*,

and

.. math::

     M= A \otimes I - h\, B \otimes J.

Coming from the discretization of

.. math::

     \begin{array}{ll}
     {\displaystyle
     \frac{{\rm d} {\bf y}(t)}{{\rm d}t}}=J {\bf y}(t)
     + {\bf g}(t), & t \in (t_0, T],\\
     \\
     {\bf y}(t_0) = {\bf z}, &  \\
     \end{array}

wiht **Boundary Value Methods**.

Library
=======

To assemble the linear system the key routine is represented by the method
:meth:`bvmatrices.builder.mab`.

.. automodule:: bvmatrices.builder
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Bibliography
============

Preconditioners and material used in this library are discussed in detail in a
number of papers:

.. bibliography::
   :all:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
