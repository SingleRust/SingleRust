System Requirements
=========================================

This page outlines the requirements for installing and using SingleRust, including both Rust itself and system dependencies.

Rust Installation
----------------------------

SingleRust requires Rust to be installed on your system. We recommend using the latest stable version.

**Installing Rust with rustup (recommended):**

.. code-block:: bash

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Follow the on-screen instructions to complete the installation. This will install Rust and Cargo (Rust's package manager).

**Verifying your installation:**

.. code-block:: bash

   rustc --version
   cargo --version

Both commands should display version information if Rust is installed correctly.

System Dependencies
----------------------------

SingleRust requires several system libraries to function properly:

1. **pkg-config**: A helper tool for compiling applications and libraries
2. **fontconfig**: A library for configuring and customizing font access
3. **zlib**: A compression library
4. **HDF5**: A data model, library, and file format for storing and managing data
5. **OpenBLAS**: (Optional) - An optimized BLAS library for linear algebra operations

Installation by Platform
----------------------------

Ubuntu/Debian
~~~~~~~~~~~~~~~~~
.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       pkg-config \
       libfontconfig1-dev \
       libhdf5-dev \
       zlib1g-dev \
       libopenblas-dev

Fedora/RHEL/CentOS
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo dnf install -y \
       pkgconfig \
       fontconfig-devel \
       hdf5-devel \
       zlib-devel \
       openblas-devel

Arch Linux
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sudo pacman -S \
       pkg-config \
       fontconfig \
       hdf5 \
       zlib \
       openblas

macOS
~~~~~~~~~~~~~~~~~

Using Homebrew:

.. code-block:: bash

   brew install \
       pkg-config \
       fontconfig \
       hdf5 \
       zlib \
       openblas

Windows
~~~~~~~~~~~~~~~~~

Using MSYS2:

.. code-block:: bash

   pacman -S \
       mingw-w64-x86_64-pkg-config \
       mingw-w64-x86_64-fontconfig \
       mingw-w64-x86_64-hdf5 \
       mingw-w64-x86_64-zlib \
       mingw-w64-x86_64-openblas

Alternatively, you can use Conda/Mamba to install these dependencies in a cross-platform way:

.. code-block:: bash

   conda install -c conda-forge \
       pkg-config \
       fontconfig \
       hdf5 \
       zlib \
       openblas

Verifying Dependencies
----------------------------

You can verify that the HDF5 library is correctly installed by running:

.. code-block:: bash

   pkg-config --modversion hdf5

This should output the version of HDF5 installed on your system.

Troubleshooting Common Issues
----------------------------

HDF5 Not Found
~~~~~~~~~~~~~~~~~

If you encounter an error like `Package hdf5 was not found in the pkg-config search path`:

1. Ensure HDF5 is installed with development headers
2. Set the PKG_CONFIG_PATH environment variable:

   .. code-block:: bash
      
      # On Linux/macOS
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/pkgconfig
      
      # On Windows with MSYS2
      export PKG_CONFIG_PATH=/mingw64/lib/pkgconfig

Fontconfig Issues
~~~~~~~~~~~~~~~~~

If you encounter fontconfig-related errors:

1. Make sure fontconfig development files are installed
2. On some systems, you may need to run:

   .. code-block:: bash
      
      # Regenerate fontconfig cache
      fc-cache -f -v

Linker Errors
~~~~~~~~~~~~~~~~~

If you encounter linker errors when building SingleRust:

1. Ensure that library paths are correctly set:

   .. code-block:: bash
      
      # On Linux
      export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
      
      # On macOS
      export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

2. Make sure your Rust compiler can find the system libraries

Next Steps
----------------------------

Once you have Rust and all system dependencies installed, proceed to the :doc:`installation` page for instructions on installing SingleRust itself.