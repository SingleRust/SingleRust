IO Module
=========

The IO module provides functions for reading and writing single-cell data.

Enums
-----

FileScope
~~~~~~~~

.. code-block:: rust

   pub enum FileScope {
       Read = 0,
       ReadWrite = 1,
   }

The ``FileScope`` enum specifies the access mode for files:

- ``FileScope::Read``: Open file in read-only mode
- ``FileScope::ReadWrite``: Open file in read-write mode

Functions
--------

read_h5ad
~~~~~~~~

.. code-block:: rust

   pub fn read_h5ad<P: AsRef<Path>>(
       path_to_file: P,
       scope: FileScope,
       enable_cache: bool
   ) -> anyhow::Result<AnnData<H5>>

Reads an H5AD file into an ``AnnData<H5>`` object.

Parameters:
- ``path_to_file``: Path to the H5AD file
- ``scope``: File access mode (``FileScope::Read`` or ``FileScope::ReadWrite``)
- ``enable_cache``: Whether to enable caching of data

Returns:
- ``anyhow::Result<AnnData<H5>>``: The loaded AnnData object or an error

Example:

.. code-block:: rust

   use single_rust::io::{read_h5ad, FileScope};

   // Read an H5AD file in read-only mode without caching
   let adata = read_h5ad("path/to/file.h5ad", FileScope::Read, false)?;

read_h5ad_memory
~~~~~~~~~~~~~~~

.. code-block:: rust

   pub fn read_h5ad_memory<P: AsRef<Path>>(path_to_file: P) -> anyhow::Result<IMAnnData>

Reads an H5AD file into an in-memory ``IMAnnData`` object.

Parameters:
- ``path_to_file``: Path to the H5AD file

Returns:
- ``anyhow::Result<IMAnnData>``: The loaded in-memory AnnData object or an error

Example:

.. code-block:: rust

   use single_rust::io::read_h5ad_memory;

   // Read an H5AD file into memory
   let adata = read_h5ad_memory("path/to/file.h5ad")?;
   println!("Loaded {} cells and {} genes", adata.n_obs(), adata.n_vars());

Data Types
---------

AnnData<H5>
~~~~~~~~~~

The ``AnnData<H5>`` type represents an AnnData object backed by an H5 file. It provides access to the data in the H5AD file without loading everything into memory.

IMAnnData
~~~~~~~~

The ``IMAnnData`` type represents an in-memory AnnData object. All data is loaded into memory, which can be faster for operations but requires more RAM.

Best Practices
-------------

File Handling
~~~~~~~~~~~

1. Use ``read_h5ad`` with ``FileScope::Read`` for large datasets when you only need to read data
2. Use ``read_h5ad`` with ``FileScope::ReadWrite`` when you need to modify the data and save it back
3. Use ``read_h5ad_memory`` for smaller datasets or when you need to perform many operations on the data

Memory Management
~~~~~~~~~~~~~~~

For very large datasets, consider:

1. Using chunked operations (``ComputationMode::Chunked``) to process data in smaller pieces
2. Using H5-backed objects (``AnnData<H5>``) instead of in-memory objects (``IMAnnData``)
3. Filtering cells and genes early in your pipeline to reduce memory usage