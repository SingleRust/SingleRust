Core API
========

This section documents the core types and functions provided by SingleRust.

Enums
-----

Direction
~~~~~~~~

.. code-block:: rust

   pub enum Direction {
       Row = 0,
       Column = 1,
   }

The ``Direction`` enum specifies the axis for operations on matrices:

- ``Direction::Row``: Operations are applied along rows (cells)
- ``Direction::Column``: Operations are applied along columns (genes)

Methods:

- ``is_row() -> bool``: Returns true if the direction is ``Row``

Example:

.. code-block:: rust

   use single_rust::Direction;

   let direction = Direction::Row;
   assert!(direction.is_row());

FlexValue
~~~~~~~~

.. code-block:: rust

   pub enum FlexValue {
       Absolute(f32),
       Relative(f32),
       None,
   }

The ``FlexValue`` enum provides flexible parameter specification:

- ``FlexValue::Absolute(value)``: An absolute value
- ``FlexValue::Relative(value)``: A relative value, typically in the range [0.0, 1.0]
- ``FlexValue::None``: No value specified

Methods:

- ``is_absolute() -> bool``: Returns true if the value is ``Absolute``
- ``is_relative() -> bool``: Returns true if the value is ``Relative``
- ``is_none() -> bool``: Returns true if the value is ``None``
- ``is_some() -> bool``: Returns true if the value is not ``None``

Precision
~~~~~~~~

.. code-block:: rust

   pub enum Precision {
       Single,
       Double,
   }

The ``Precision`` enum specifies the floating-point precision:

- ``Precision::Single``: Single precision (32-bit, ``f32``)
- ``Precision::Double``: Double precision (64-bit, ``f64``)

The default precision is ``Precision::Single``.

FeatureSelection
~~~~~~~~~~~~~~

.. code-block:: rust

   pub enum FeatureSelection {
       HighlyVariableCol(String),
       HighlyVariable(usize),
       Randomized(usize),
       VarianceThreshold(f64),
       None,
   }

The ``FeatureSelection`` enum specifies methods for selecting features (genes):

- ``HighlyVariableCol(String)``: Select genes based on a column in the var dataframe
- ``HighlyVariable(usize)``: Select the top N highly variable genes
- ``Randomized(usize)``: Select N random genes
- ``VarianceThreshold(f64)``: Select genes with variance above a threshold
- ``None``: No feature selection

ComputationMode
~~~~~~~~~~~~~

.. code-block:: rust

   pub enum ComputationMode {
       Chunked(usize),
       Whole,
   }

The ``ComputationMode`` enum specifies how computations are performed:

- ``ComputationMode::Chunked(chunk_size)``: Process data in chunks of the specified size
- ``ComputationMode::Whole``: Process the entire dataset at once

Traits
------

ComputeNonZero
~~~~~~~~~~~~

.. code-block:: rust

   pub trait ComputeNonZero {
       fn nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
       where
           T: PrimInt + Unsigned + Zero + AddAssign;

       fn nonzero_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
       where
           T: PrimInt + Unsigned + Zero + AddAssign;
   }

The ``ComputeNonZero`` trait provides methods for counting non-zero elements:

- ``nonzero_whole<T>(...)``: Counts non-zero elements along the specified direction
- ``nonzero_chunk<T>(...)``: Counts non-zero elements in chunks

ComputeSum
~~~~~~~~~

.. code-block:: rust

   pub trait ComputeSum {
       fn sum_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
       where
           T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

       fn sum_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
       where
           T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;
   }

The ``ComputeSum`` trait provides methods for summing elements:

- ``sum_whole<T>(...)``: Sums elements along the specified direction
- ``sum_chunk<T>(...)``: Sums elements in chunks

ComputeVariance
~~~~~~~~~~~~~

.. code-block:: rust

   pub trait ComputeVariance {
       fn variance_whole<I, T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
       where
           I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
           T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

       fn variance_chunk<I, T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
       where
           I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
           T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;
   }

The ``ComputeVariance`` trait provides methods for calculating variance:

- ``variance_whole<I, T>(...)``: Calculates variance along the specified direction
- ``variance_chunk<I, T>(...)``: Calculates variance in chunks

ComputeMinMax
~~~~~~~~~~~

.. code-block:: rust

   pub trait ComputeMinMax {
       fn min_max_whole<T>(&self, direction: &Direction) -> anyhow::Result<(Vec<T>, Vec<T>)>
       where
           T: NumCast + Copy + PartialOrd + NumericOps;

       fn min_max_chunk<T>(&self, direction: &Direction, reference: (&mut Vec<T>, &mut Vec<T>)) -> anyhow::Result<()>
       where
           T: NumCast + Copy + PartialOrd + NumericOps;
   }

The ``ComputeMinMax`` trait provides methods for finding minimum and maximum values:

- ``min_max_whole<T>(...)``: Finds min and max values along the specified direction
- ``min_max_chunk<T>(...)``: Finds min and max values in chunks

Functions
--------

convert_to_array_f64
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub fn convert_to_array_f64(arr_data: &ArrayData) -> anyhow::Result<Array2<f64>>

Converts various array types to a dense ``ndarray::Array2<f64>`` array.

Example:

.. code-block:: rust

   use single_rust::convert_to_array_f64;

   // Convert an ArrayData to a dense Array2<f64>
   let dense_array = convert_to_array_f64(&adata.x().get_inner())?;