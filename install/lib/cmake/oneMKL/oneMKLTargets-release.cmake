#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MKL::onemkl_rng_mklcpu" for configuration "Release"
set_property(TARGET MKL::onemkl_rng_mklcpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MKL::onemkl_rng_mklcpu PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonemkl_rng_mklcpu.so.0"
  IMPORTED_SONAME_RELEASE "libonemkl_rng_mklcpu.so.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS MKL::onemkl_rng_mklcpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_MKL::onemkl_rng_mklcpu "${_IMPORT_PREFIX}/lib/libonemkl_rng_mklcpu.so.0" )

# Import target "MKL::onemkl_rng_mklgpu" for configuration "Release"
set_property(TARGET MKL::onemkl_rng_mklgpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MKL::onemkl_rng_mklgpu PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonemkl_rng_mklgpu.so.0"
  IMPORTED_SONAME_RELEASE "libonemkl_rng_mklgpu.so.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS MKL::onemkl_rng_mklgpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_MKL::onemkl_rng_mklgpu "${_IMPORT_PREFIX}/lib/libonemkl_rng_mklgpu.so.0" )

# Import target "MKL::onemkl" for configuration "Release"
set_property(TARGET MKL::onemkl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MKL::onemkl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libonemkl.so.0"
  IMPORTED_SONAME_RELEASE "libonemkl.so.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS MKL::onemkl )
list(APPEND _IMPORT_CHECK_FILES_FOR_MKL::onemkl "${_IMPORT_PREFIX}/lib/libonemkl.so.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
