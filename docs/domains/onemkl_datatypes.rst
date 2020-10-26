.. _onemkl_datatypes:

oneMKL defined datatypes
------------------------

oneMKL dense and sparse linear algebra routines use scoped enum types as type-safe replacements for the traditional character arguments used in C/Fortran implementations of BLAS and LAPACK. These types all belong to the ``oneapi::mkl`` namespace.  

Each enumeration value comes with two names: A single-character name (the traditional BLAS/LAPACK character) and a longer, more descriptive name. The two names are exactly equivalent and may be used interchangeably.

      .. _onemkl_enum_transpose:

      .. rubric:: transpose
         :name: transpose
         :class: sectiontitle

      The ``transpose`` type specifies whether an input matrix should be
      transposed and/or conjugated. It can take the following values:

      .. container:: tablenoborder

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

      .. _onemkl_enum_uplo:

      .. rubric:: uplo
         :name: uplo
         :class: sectiontitle

      The ``uplo`` type specifies whether the lower or upper triangle of
      a triangular, symmetric, or Hermitian matrix should be accessed.
      It can take the following values:

      .. container:: tablenoborder

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

      In both cases, elements that are not in the selected triangle are
      not accessed or updated.

      .. _onemkl_enum_diag:

      .. rubric:: diag
         :name: diag
         :class: sectiontitle

      The ``diag`` type specifies the values on the diagonal of a
      triangular matrix. It can take the following values:

      .. container:: tablenoborder

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
              -  The matrix is unit triangular (the diagonal entries are all 1's). The diagonal entries in the matrix data are not accessed.

      .. _onemkl_enum_side:

      .. rubric:: side
         :name: side
         :class: sectiontitle

      The ``side`` type specifies the order of matrix multiplication
      when one matrix has a special form (triangular, symmetric, or
      Hermitian):

      .. container:: tablenoborder

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

      .. _onemkl_enum_offset:

      .. rubric:: offset
         :name: offset
         :class: sectiontitle

      The ``offset`` type specifies whether the offset to apply to an
      output matrix is a fix offset, column offset or row offset. It can
      take the following values

      .. container:: tablenoborder

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

      .. _onemkl_enum_index_base:

      .. rubric:: index_base
         :name: index_base
         :class: sectiontitle

      The ``index_base`` type specifies how values in index arrays are interpreted. For instance, a sparse matrix stores nonzero values and the
      indices that they correspond to.  The indices are traditionally provided in one of two forms: C/C++-style using zero-based
      indices, or Fortran-style using one-based indices. The ``index_base`` type can take the following values:

      .. container:: tablenoborder

         .. list-table::
            :header-rows: 1

            * -  Name
              -  Description
            * -  ``index_base::zero``
              -  Index arrays for an input matrix are provided using zero-based (C/C++ style) index values.  That is, indices start at 0.
            * -  ``index_base::one``
              -  Index arrays for an input matrix are provided using one-based (Fortran style) index values.  That is, indices start at 1.


.. note::
        :ref:`onemkl_appendix` may contain other API design decisions or recommendations that may be of use to the general developer of oneMKL, but which may not necessarily be part of the oneMKL specification.


