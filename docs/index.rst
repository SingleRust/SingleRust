SingleRust 🧬 Documentation
=========================================

.. image:: https://readthedocs.org/projects/singlerust/badge/?version=latest
   :target: https://singlerust.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Welcome to SingleRust's documentation, a pioneering library for the Rust programming language, focused on production-grade, high-throughput analysis pipelines for single-cell data.

.. note::
    This documentation is currently very much WIP. This is currently a highly unstable version!

.. note::
   SingleRust is currently in early development stage. The API may change as the library evolves.

Introduction
------------------

SingleRust is designed to leverage Rust's fearless concurrency model to transition single-cell data analysis from initial prototyping to robust, scalable deployments. The library aims to provide a comprehensive toolkit for single-cell analysis with a focus on performance and reliability.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   requirements
   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/import_export
   user_guide/preprocessing
   user_guide/normalization
   user_guide/dimension_reduction
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/io
   api/memory
   api/statistics
   api/processing
   api/plot

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   code_of_conduct
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`