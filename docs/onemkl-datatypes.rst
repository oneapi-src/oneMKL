.. _onemkl_datatypes:

oneMKL Defined Datatypes
========================


oneMKL BLAS and LAPACK for Data Parallel C++ (DPC++) introduces
several new enumeration data types, which are type-safe versions of
the traditional Fortran characters in BLAS and LAPACK. They are
declared in ``types.hpp``, which is included automatically when
you include ``mkl.hpp``. Like all oneMKL DPC++ functionality, they belong to the namespace ``oneapi::mkl``.


Each enumeration value comes with two names: A single-character name
(the traditional BLAS/LAPACK character) and a longer, descriptive
name. The two names are exactly equivalent and may be used
interchangeably.


transpose
---------

The ``transpose`` type specifies whether an input matrix should be
transposed and/or conjugated. It can take the following values:


.. list-table::
   :header-rows: 1

   * -  Short Name
     -  Long Name
     -  Description
   * -  ``transpose::N``
     -  ``transpose::nontrans``
     -  Do not transpose or conjugate the matrix.
   * -  ``transpose::T``
     -  ``transpose::trans``
     -  Transpose the matrix.
   * -  ``transpose::C``
     -  ``transpose::conjtrans``
     -  Perform Hermitian transpose (transpose and conjugate). Only applicable to complex matrices.




uplo
----

The ``uplo`` type specifies whether the lower or upper triangle of a riangular, symmetric, or Hermitian matrix should be accessed.

It can take the following values:


.. list-table::
   :header-rows: 1

   * -  Short Name
     -  Long Name
     -  Description
   * -  ``uplo::U``
     -  ``uplo::upper``
     -  Access the upper triangle of the matrix.
   * -  ``uplo::L``
     -  ``uplo::lower``
     -  Access the lower triangle of the matrix.




In both cases, elements that are not in the selected triangle are not accessed or updated.


diag
----


The ``diag`` type specifies the values on the diagonal of a triangular matrix. It can take the following values:


.. list-table::
   :header-rows: 1

   * -  Short Name
     -  Long Name
     -  Description
   * -  ``diag::N``
     -  ``diag::nonunit``
     -  The matrix is not unit triangular. The diagonal entries are stored with the matrix data.
   * -  ``diag::U``
     -  ``diag::unit``
     -  The matrix is unit triangular (the diagonal entries are all 1s). The diagonal entries in the matrix data are not accessed.




side
----


The ``side`` type specifies the order of matrix multiplication when one matrix has a special form (triangular, symmetric, or Hermitian):


.. list-table::
   :header-rows: 1

   * -  Short Name
     -  Long Name
     -  Description
   * -  ``side::L``
     -  ``side::left``
     -  The special form matrix is on the left in the multiplication.
   * -  ``side::R``
     -  ``side::right``
     -  The special form matrix is on the right in the multiplication.


offset
------


The ``offset`` type specifies whether the offset to apply to an output matrix is a fix offset, column offset or row offset. It can take the following values


.. list-table::
   :header-rows: 1

   * -  Short Name
     -  Long Name
     -  Description
   * -  ``offset::F``
     -  ``offset::fix``
     -  The offset to apply to the output matrix is fix, all the inputs in the ``C_offset`` matrix has the same value given by the first element in the ``co`` array.
   * -  ``offset::C``
     -  ``offset::column``
     -  The offset to apply to the output matrix is a column offset, that is to say all the columns in the ``C_offset`` matrix are the same and given by the elements in the ``co`` array.
   * -  ``offset::R``
     -  ``offset::row``
     -  The offset to apply to the output matrix is a row offset, that is to say all the rows in the ``C_offset`` matrix are the same and given by the elements in the ``co`` array.

**Parent topic:** :ref:`onemkl`
