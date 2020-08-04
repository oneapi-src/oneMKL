.. _onemkl_datatypes:

oneMKL defined datatypes
========================


.. container::


   oneMKL BLAS and LAPACK for Data Parallel C++ (DPC++) introduces
   several new enumeration data types, which are type-safe versions of
   the traditional Fortran characters in BLAS and LAPACK. They are
   declared in ``types.hpp``, which is included automatically when
   you include ``mkl.hpp``. Like
   all oneMKL DPC++ functionality, they belong to the
   namespace ``oneapi::mkl``.


   Each enumeration value comes with two names: A single-character name
   (the traditional BLAS/LAPACK character) and a longer, descriptive
   name. The two names are exactly equivalent and may be used
   interchangeably.


   .. container:: section
      :name: GUID-97715A19-7DDE-4738-9E7A-53554E5B702B


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
              -  Perform Hermitian transpose (transpose and conjugate). Only          applicable to complex matrices.




   .. container:: section
      :name: GUID-DD566CC1-62E5-4AF1-A407-FB4E979B753D


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


   .. container:: section
      :name: GUID-419CF945-4868-400D-B05C-50ABABD73961


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
              -  The matrix is not unit triangular. The diagonal entries are          stored with the matrix data.
            * -  ``diag::U`` 
              -  ``diag::unit`` 
              -  The matrix is unit triangular (the diagonal entries are all          1s). The diagonal entries in the matrix data are not accessed.




   .. container:: section
      :name: GUID-538307BC-A47D-4290-B5B4-CB54CFB25242


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
              -  The special form matrix is on the left in the          multiplication.
            * -  ``side::R`` 
              -  ``side::right`` 
              -  The special form matrix is on the right in the          multiplication.




   .. container:: section
      :name: GUID-D25C1BB5-81B8-4591-A815-C881B59E7C5B


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
              -  The offset to apply to the output matrix is fix, all the          inputs in the ``C_offset`` matrix has the same value given by         the first element in the ``co`` array.
            * -  ``offset::C`` 
              -  ``offset::column`` 
              -  The offset to apply to the output matrix is a column          offset, that is to say all the columns in the ``C_offset``         matrix are the same and given by the elements in the ``co``         array.
            * -  ``offset::R`` 
              -  ``offset::row`` 
              -  The offset to apply to the output matrix is a row offset,          that is to say all the rows in the ``C_offset`` matrix are the         same and given by the elements in the ``co`` array.

      **Parent topic:** :ref:`onemkl`
