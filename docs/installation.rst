Installation
============

This guide will help you install SingleRust and set up your environment for single-cell data analysis.

Prerequisites
----------------------------------------

Before installing SingleRust, ensure you have installed all the required system dependencies.

- Rust (latest stable version recommended)
- Cargo (comes with Rust)
- System dependencies: pkg-config, fontconfig, zlib, HDF5, etc.

Please refer to the :doc:`requirements` page for detailed instructions on installing Rust and all necessary system dependencies for your platform.

Installing SingleRust
----------------------------

Using cargo add (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to add SingleRust to your project is using the ``cargo add`` command:

.. code-block:: bash

   # Add the latest version from crates.io
   cargo add single_rust

   # Add a specific version
   cargo add single_rust@0.2.1-alpha.1

   # Add from GitHub repository
   cargo add single_rust --git https://github.com/SingleRust/SingleRust

Manually editing Cargo.toml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can manually add SingleRust to your ``Cargo.toml`` file:

.. code-block:: toml

   [dependencies]
   # From crates.io
   single_rust = "0.2.1-alpha.1"

   # From GitHub
   single_rust = { git = "https://github.com/SingleRust/SingleRust" }

Using a local copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have downloaded or cloned the SingleRust repository locally, you can reference it in your project:

.. code-block:: toml

   [dependencies]
   # Using a local path
   single_rust = { path = "../path/to/SingleRust" }

Or with cargo add:

.. code-block:: bash

   cargo add single_rust --path ../path/to/SingleRust

Optional Features
----------------------------

SingleRust comes with several optional features:

- **lapack**: Enables LAPACK integration for numerical linear algebra
- **faer**: Enables Faer for numerical linear algebra
- **simba**: Enables SIMD acceleration for statistics

To enable these features, modify your dependency in ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   single_rust = { version = "0.2.1-alpha.1", features = ["lapack", "faer"] }

Verifying Installation
----------------------------

To verify your installation, create a new Rust project:

.. code-block:: bash

   cargo new test_singlerust
   cd test_singlerust

Add SingleRust as a dependency in your ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   single_rust = "0.2.1-alpha.1"

Create a simple test in ``src/main.rs``:

.. code-block:: rust

   use single_rust::io;

   fn main() {
       println!("SingleRust is installed correctly!");
   }

Run the test:

.. code-block:: bash

   cargo run

If everything is set up correctly, you should see the message "SingleRust is installed correctly!" printed to your console.