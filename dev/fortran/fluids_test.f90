module python_utils
  use iso_c_binding
  implicit none

  ! Python object handles
  type(c_ptr) :: py_module = c_null_ptr
  type(c_ptr) :: py_none = c_null_ptr
  
contains
  subroutine init_python()
    interface
      subroutine Py_Initialize() bind(c, name='Py_Initialize')
      end subroutine
      
      function PyImport_ImportModule(name) bind(c, name='PyImport_ImportModule')
        use iso_c_binding
        character(kind=c_char), dimension(*) :: name
        type(c_ptr) :: PyImport_ImportModule
      end function
    end interface

    ! Initialize Python
    call Py_Initialize()
    
    ! Import fluids module
    py_module = PyImport_ImportModule('fluids'//c_null_char)
    if (.not. c_associated(py_module)) then
      print *, 'Failed to import fluids module'
      call exit(1)
    end if
  end subroutine

  subroutine test_fluids()
    interface
      function PyObject_GetAttrString(obj, name) bind(c, name='PyObject_GetAttrString')
        use iso_c_binding
        type(c_ptr), value :: obj
        character(kind=c_char), dimension(*) :: name
        type(c_ptr) :: PyObject_GetAttrString
      end function
      
      function PyFloat_AsDouble(obj) bind(c, name='PyFloat_AsDouble')
        use iso_c_binding
        type(c_ptr), value :: obj
        real(c_double) :: PyFloat_AsDouble
      end function

      function PyUnicode_AsUTF8(obj) bind(c, name='PyUnicode_AsUTF8')
        use iso_c_binding
        type(c_ptr), value :: obj
        type(c_ptr) :: PyUnicode_AsUTF8
      end function
      
      function PyTuple_New(size) bind(c, name='PyTuple_New')
        use iso_c_binding
        integer(c_size_t), value :: size
        type(c_ptr) :: PyTuple_New
      end function
      
      function PyDict_New() bind(c, name='PyDict_New')
        use iso_c_binding
        type(c_ptr) :: PyDict_New
      end function
      
      function PyFloat_FromDouble(val) bind(c, name='PyFloat_FromDouble')
        use iso_c_binding
        real(c_double), value :: val
        type(c_ptr) :: PyFloat_FromDouble
      end function
      
      subroutine PyDict_SetItemString(dict, key, val) bind(c, name='PyDict_SetItemString')
        use iso_c_binding
        type(c_ptr), value :: dict
        character(kind=c_char), dimension(*) :: key
        type(c_ptr), value :: val
      end subroutine
      
      function PyObject_Call(callable, args, kwargs) bind(c, name='PyObject_Call')
        use iso_c_binding
        type(c_ptr), value :: callable
        type(c_ptr), value :: args
        type(c_ptr), value :: kwargs
        type(c_ptr) :: PyObject_Call
      end function
    end interface

    type(c_ptr) :: version, reynolds_func, args, kwargs, result, str_ptr
    real(c_double) :: re
    character(len=100), pointer :: version_str
    character(kind=c_char), pointer :: c_str(:)
    character(len=100) :: temp_str
    integer :: i

    print *, 'Running fluids tests from Fortran...'

    ! Get version
    version = PyObject_GetAttrString(py_module, '__version__'//c_null_char)
    if (c_associated(version)) then
      str_ptr = PyUnicode_AsUTF8(version)
      call c_f_pointer(str_ptr, c_str, [100])
      temp_str = ''
      i = 1
      do while (c_str(i) /= c_null_char .and. i <= 100)
        temp_str(i:i) = c_str(i)
        i = i + 1
      end do
      
      print *, '✓ Successfully imported fluids'
      print *, '✓ Fluids version: ', trim(temp_str)
    end if
    ! Test Reynolds number calculation
    reynolds_func = PyObject_GetAttrString(py_module, 'Reynolds'//c_null_char)
    if (c_associated(reynolds_func)) then
      args = PyTuple_New(0_c_size_t)
      kwargs = PyDict_New()
      
      call PyDict_SetItemString(kwargs, 'V'//c_null_char, PyFloat_FromDouble(2.5d0))
      call PyDict_SetItemString(kwargs, 'D'//c_null_char, PyFloat_FromDouble(0.1d0))
      call PyDict_SetItemString(kwargs, 'rho'//c_null_char, PyFloat_FromDouble(1000.0d0))
      call PyDict_SetItemString(kwargs, 'mu'//c_null_char, PyFloat_FromDouble(0.001d0))
      
      result = PyObject_Call(reynolds_func, args, kwargs)
      if (c_associated(result)) then
        re = PyFloat_AsDouble(result)
        print *, '✓ Reynolds number calculation successful: ', re
        if (re > 0) then
          print *, 'Assert passed: Re > 0'
        else
          print *, 'Assert failed: Re <= 0'
        end if
      end if
    end if
  end subroutine

  subroutine benchmark_fluids()
    interface
      function PyObject_GetAttrString(obj, name) bind(c, name='PyObject_GetAttrString')
        use iso_c_binding
        type(c_ptr), value :: obj
        character(kind=c_char), dimension(*) :: name
        type(c_ptr) :: PyObject_GetAttrString
      end function
      
      function PyTuple_New(size) bind(c, name='PyTuple_New')
        use iso_c_binding
        integer(c_size_t), value :: size
        type(c_ptr) :: PyTuple_New
      end function
      
      function PyDict_New() bind(c, name='PyDict_New')
        use iso_c_binding
        type(c_ptr) :: PyDict_New
      end function

      function PyFloat_FromDouble(val) bind(c, name='PyFloat_FromDouble')
        use iso_c_binding
        real(c_double), value :: val
        type(c_ptr) :: PyFloat_FromDouble
      end function
      
      subroutine PyDict_SetItemString(dict, key, val) bind(c, name='PyDict_SetItemString')
        use iso_c_binding
        type(c_ptr), value :: dict
        character(kind=c_char), dimension(*) :: key
        type(c_ptr), value :: val
      end subroutine
      
      function PyObject_Call(callable, args, kwargs) bind(c, name='PyObject_Call')
        use iso_c_binding
        type(c_ptr), value :: callable
        type(c_ptr), value :: args
        type(c_ptr), value :: kwargs
        type(c_ptr) :: PyObject_Call
      end function
    end interface

    type(c_ptr) :: friction_func
    type(c_ptr) :: args, kwargs, result
    integer :: i
    real(c_double) :: start_time, end_time
    
    print *, 'Running benchmarks:'
    
    ! Get friction_factor function
    friction_func = PyObject_GetAttrString(py_module, 'friction_factor'//c_null_char)
    if (.not. c_associated(friction_func)) then
      print *, 'Failed to get friction_factor function'
      return
    end if
    
    ! Benchmark friction_factor
    print *, 'Benchmarking friction_factor:'
    call cpu_time(start_time)
    
    do i = 1, 1000000
      args = PyTuple_New(0_c_size_t)
      kwargs = PyDict_New()
      
      call PyDict_SetItemString(kwargs, 'Re'//c_null_char, PyFloat_FromDouble(1.0d5))
      call PyDict_SetItemString(kwargs, 'eD'//c_null_char, PyFloat_FromDouble(0.0001d0))
      
      result = PyObject_Call(friction_func, args, kwargs)
    end do
    
    call cpu_time(end_time)
    print *, 'Time for 1e6 friction_factor calls: ', end_time - start_time, ' seconds'
    print *, 'Average time per call: ', (end_time - start_time) / 1000000.0, ' seconds'
  end subroutine


  subroutine test_tank()
    interface
      function PyObject_GetAttrString(obj, name) bind(c, name='PyObject_GetAttrString')
        use iso_c_binding
        type(c_ptr), value :: obj
        character(kind=c_char), dimension(*) :: name
        type(c_ptr) :: PyObject_GetAttrString
      end function
      
      function PyTuple_New(size) bind(c, name='PyTuple_New')
        use iso_c_binding
        integer(c_size_t), value :: size
        type(c_ptr) :: PyTuple_New
      end function
      
      function PyDict_New() bind(c, name='PyDict_New')
        use iso_c_binding
        type(c_ptr) :: PyDict_New
      end function

      function PyFloat_FromDouble(val) bind(c, name='PyFloat_FromDouble')
        use iso_c_binding
        real(c_double), value :: val
        type(c_ptr) :: PyFloat_FromDouble
      end function
      
      function PyBool_FromLong(val) bind(c, name='PyBool_FromLong')
        use iso_c_binding
        integer(c_long), value :: val
        type(c_ptr) :: PyBool_FromLong
      end function
      
      function PyFloat_AsDouble(obj) bind(c, name='PyFloat_AsDouble')
        use iso_c_binding
        type(c_ptr), value :: obj
        real(c_double) :: PyFloat_AsDouble
      end function

      function PyUnicode_FromString(str) bind(c, name='PyUnicode_FromString')
        use iso_c_binding
        character(kind=c_char), dimension(*) :: str
        type(c_ptr) :: PyUnicode_FromString
      end function
      
      subroutine PyDict_SetItemString(dict, key, val) bind(c, name='PyDict_SetItemString')
        use iso_c_binding
        type(c_ptr), value :: dict
        character(kind=c_char), dimension(*) :: key
        type(c_ptr), value :: val
      end subroutine
      
      function PyObject_Call(callable, args, kwargs) bind(c, name='PyObject_Call')
        use iso_c_binding
        type(c_ptr), value :: callable
        type(c_ptr), value :: args
        type(c_ptr), value :: kwargs
        type(c_ptr) :: PyObject_Call
      end function

      subroutine PyTuple_SetItem(tup, pos, item) bind(c, name='PyTuple_SetItem')
        use iso_c_binding
        type(c_ptr), value :: tup
        integer(c_size_t), value :: pos
        type(c_ptr), value :: item
      end subroutine

      function PyUnicode_AsUTF8(obj) bind(c, name='PyUnicode_AsUTF8')
        use iso_c_binding
        type(c_ptr), value :: obj
        type(c_ptr) :: PyUnicode_AsUTF8
      end function

      function PyObject_CallMethod(obj, method, format) bind(c, name='PyObject_CallMethod')
        use iso_c_binding
        type(c_ptr), value :: obj
        character(kind=c_char), dimension(*) :: method
        character(kind=c_char), dimension(*) :: format
        type(c_ptr) :: PyObject_CallMethod
      end function

      function PyObject_Str(obj) bind(c, name='PyObject_Str')
        use iso_c_binding
        type(c_ptr), value :: obj
        type(c_ptr) :: PyObject_Str
      end function

      subroutine Py_DecRef(obj) bind(c, name='Py_DecRef')
        use iso_c_binding
        type(c_ptr), value :: obj
      end subroutine

    end interface

    type(c_ptr) :: tank_class, args1, kwargs1, T1, length, diameter
    type(c_ptr) :: tank_ellip, args_ellip, kwargs_ellip, ellip_length
    type(c_ptr) :: args2, kwargs2, DIN, h_max, tank_str
    type(c_ptr) :: h_from_V, V_from_h, SA_from_h
    type(c_ptr) :: arg40, arg41, arg21, result40, result41, result21
    real(c_double) :: tank_length, tank_diameter, ellip_l, max_height
    real(c_double) :: height_40, volume_41, surface_area_21
    character(kind=c_char), pointer :: c_str(:)
    character(len=1000) :: temp_str
    integer :: i
    type(c_ptr) :: str_ptr

    print *, 'Testing tank calculations:'

    ! Get TANK class
    tank_class = PyObject_GetAttrString(py_module, 'TANK'//c_null_char)
    if (.not. c_associated(tank_class)) then
      print *, 'Failed to get TANK class'
      return
    end if

    ! Test basic tank creation
    args1 = PyTuple_New(0_c_size_t)
    kwargs1 = PyDict_New()
    
    call PyDict_SetItemString(kwargs1, 'V'//c_null_char, PyFloat_FromDouble(10.0d0))
    call PyDict_SetItemString(kwargs1, 'L_over_D'//c_null_char, PyFloat_FromDouble(0.7d0))
    call PyDict_SetItemString(kwargs1, 'sideB'//c_null_char, PyUnicode_FromString('conical'//c_null_char))
    call PyDict_SetItemString(kwargs1, 'horizontal'//c_null_char, PyBool_FromLong(0_c_long))

    T1 = PyObject_Call(tank_class, args1, kwargs1)
    if (c_associated(T1)) then
      length = PyObject_GetAttrString(T1, 'L'//c_null_char)
      diameter = PyObject_GetAttrString(T1, 'D'//c_null_char)
      
      if (c_associated(length) .and. c_associated(diameter)) then
        tank_length = PyFloat_AsDouble(length)
        tank_diameter = PyFloat_AsDouble(diameter)
        print *, '✓ Tank length: ', tank_length
        print *, '✓ Tank diameter: ', tank_diameter
      end if
    end if

    ! Test ellipsoidal tank
    args_ellip = PyTuple_New(0_c_size_t)
    kwargs_ellip = PyDict_New()
    
    call PyDict_SetItemString(kwargs_ellip, 'D'//c_null_char, PyFloat_FromDouble(10.0d0))
    call PyDict_SetItemString(kwargs_ellip, 'V'//c_null_char, PyFloat_FromDouble(500.0d0))
    call PyDict_SetItemString(kwargs_ellip, 'horizontal'//c_null_char, PyBool_FromLong(0_c_long))
    call PyDict_SetItemString(kwargs_ellip, 'sideA'//c_null_char, PyUnicode_FromString('ellipsoidal'//c_null_char))
    call PyDict_SetItemString(kwargs_ellip, 'sideB'//c_null_char, PyUnicode_FromString('ellipsoidal'//c_null_char))
    call PyDict_SetItemString(kwargs_ellip, 'sideA_a'//c_null_char, PyFloat_FromDouble(1.0d0))
    call PyDict_SetItemString(kwargs_ellip, 'sideB_a'//c_null_char, PyFloat_FromDouble(1.0d0))

    tank_ellip = PyObject_Call(tank_class, args_ellip, kwargs_ellip)
    if (c_associated(tank_ellip)) then
      ellip_length = PyObject_GetAttrString(tank_ellip, 'L'//c_null_char)
      if (c_associated(ellip_length)) then
        ellip_l = PyFloat_AsDouble(ellip_length)
        print *, '✓ Ellipsoidal tank L: ', ellip_l
      end if
    end if

    ! Test torispherical tank
    args2 = PyTuple_New(0_c_size_t)
    kwargs2 = PyDict_New()
    
    call PyDict_SetItemString(kwargs2, 'L'//c_null_char, PyFloat_FromDouble(3.0d0))
    call PyDict_SetItemString(kwargs2, 'D'//c_null_char, PyFloat_FromDouble(5.0d0))
    call PyDict_SetItemString(kwargs2, 'horizontal'//c_null_char, PyBool_FromLong(0_c_long))
    call PyDict_SetItemString(kwargs2, 'sideA'//c_null_char, PyUnicode_FromString('torispherical'//c_null_char))
    call PyDict_SetItemString(kwargs2, 'sideB'//c_null_char, PyUnicode_FromString('torispherical'//c_null_char))
    call PyDict_SetItemString(kwargs2, 'sideA_f'//c_null_char, PyFloat_FromDouble(1.0d0))
    call PyDict_SetItemString(kwargs2, 'sideA_k'//c_null_char, PyFloat_FromDouble(0.1d0))
    call PyDict_SetItemString(kwargs2, 'sideB_f'//c_null_char, PyFloat_FromDouble(1.0d0))
    call PyDict_SetItemString(kwargs2, 'sideB_k'//c_null_char, PyFloat_FromDouble(0.1d0))

    DIN = PyObject_Call(tank_class, args2, kwargs2)
    if (c_associated(DIN)) then
      ! Get string representation
      tank_str = PyObject_Str(DIN)
      if (c_associated(tank_str)) then
        str_ptr = PyUnicode_AsUTF8(tank_str)
        if (c_associated(str_ptr)) then
          call c_f_pointer(str_ptr, c_str, [1000])
          temp_str = ''
          i = 1
          do while (c_str(i) /= c_null_char .and. i <= 1000)
            temp_str(i:i) = c_str(i)
            i = i + 1
          end do
          print *, '✓ Tank representation: ', trim(temp_str)
        end if
      end if

      ! Get max height
      h_max = PyObject_GetAttrString(DIN, 'h_max'//c_null_char)
      if (c_associated(h_max)) then
        max_height = PyFloat_AsDouble(h_max)
        print *, '✓ Tank max height: ', max_height
      end if

      ! Test h_from_V method
      h_from_V = PyObject_GetAttrString(DIN, 'h_from_V'//c_null_char)
      if (c_associated(h_from_V)) then
        ! Create tuple argument
        arg40 = PyTuple_New(1_c_size_t)
        ! Set tuple item directly
        call PyTuple_SetItem(arg40, 0_c_size_t, PyFloat_FromDouble(40.0d0))
        ! Call method
        result40 = PyObject_Call(h_from_V, arg40, c_null_ptr)
        if (c_associated(result40)) then
          height_40 = PyFloat_AsDouble(result40)
          print *, '✓ Height at V=40: ', height_40

          ! Cleanup
          if (c_associated(result40)) then
            call Py_DecRef(result40)
          end if
        end if

        ! Cleanup
        if (c_associated(h_from_V)) then
          call Py_DecRef(h_from_V)
        end if
        if (c_associated(arg40)) then
          call Py_DecRef(arg40)
        end if
      end if



      ! Test SA_from_h method
      SA_from_h = PyObject_GetAttrString(DIN, 'SA_from_h'//c_null_char)
      if (c_associated(SA_from_h)) then
        ! Create tuple argument
        arg21 = PyTuple_New(1_c_size_t)
        ! Set tuple item directly
        call PyTuple_SetItem(arg21, 0_c_size_t, PyFloat_FromDouble(2.1d0))
        ! Call method
        result21 = PyObject_Call(SA_from_h, arg21, c_null_ptr)
        if (c_associated(result21)) then
          surface_area_21 = PyFloat_AsDouble(result21)
          print *, '✓ Surface area at h=2.1: ', surface_area_21

          ! Cleanup
          if (c_associated(result21)) then
            call Py_DecRef(result21)
          end if
        end if

        ! Cleanup
        if (c_associated(SA_from_h)) then
          call Py_DecRef(SA_from_h)
        end if
        if (c_associated(arg21)) then
          call Py_DecRef(arg21)
        end if
      end if
    end if

  end subroutine test_tank
  ! Helper function to convert C string pointer to Fortran string
  subroutine c_f_string_ptr(c_str, f_str)
    type(c_ptr), intent(in) :: c_str
    character(len=*), pointer, intent(out) :: f_str
    
    character(kind=c_char), pointer :: tmp(:)
    integer :: i, n
    character(len=1000) :: temp_str  ! Temporary buffer
    
    call c_f_pointer(c_str, tmp, [1000])  ! Limited to reasonable size
    n = 0
    do while (tmp(n+1) /= c_null_char .and. n < 1000)
      n = n + 1
      temp_str(n:n) = tmp(n)
    end do
    
    allocate(character(len=n) :: f_str)
    f_str = temp_str(1:n)
  end subroutine

subroutine test_reynolds()
    use iso_c_binding
    implicit none

    interface
      function PyObject_GetAttrString(obj, name) bind(c, name='PyObject_GetAttrString')
        import :: c_ptr, c_char
        type(c_ptr), value :: obj
        character(kind=c_char), dimension(*) :: name
        type(c_ptr) :: PyObject_GetAttrString
      end function
      
      function PyTuple_New(size) bind(c, name='PyTuple_New')
        import :: c_ptr, c_size_t
        integer(c_size_t), value :: size
        type(c_ptr) :: PyTuple_New
      end function
      
      function PyDict_New() bind(c, name='PyDict_New')
        import :: c_ptr
        type(c_ptr) :: PyDict_New
      end function

      function PyFloat_FromDouble(val) bind(c, name='PyFloat_FromDouble')
        import :: c_ptr, c_double
        real(c_double), value :: val
        type(c_ptr) :: PyFloat_FromDouble
      end function
      
      function PyFloat_AsDouble(obj) bind(c, name='PyFloat_AsDouble')
        import :: c_ptr, c_double
        type(c_ptr), value :: obj
        real(c_double) :: PyFloat_AsDouble
      end function
      
      subroutine PyDict_SetItemString(dict, key, val) bind(c, name='PyDict_SetItemString')
        import :: c_ptr, c_char
        type(c_ptr), value :: dict
        character(kind=c_char), dimension(*) :: key
        type(c_ptr), value :: val
      end subroutine
      
      function PyObject_Call(callable, args, kwargs) bind(c, name='PyObject_Call')
        import :: c_ptr
        type(c_ptr), value :: callable
        type(c_ptr), value :: args
        type(c_ptr), value :: kwargs
        type(c_ptr) :: PyObject_Call
      end function
    end interface

    type(c_ptr) :: reynolds_func, args, kwargs, result
    real(c_double) :: Re1, Re2
    logical :: test_passed

    print *, 'Testing Reynolds number calculations:'

    ! Get Reynolds function
    reynolds_func = PyObject_GetAttrString(py_module, "Reynolds"//C_NULL_CHAR)
    if (.not. c_associated(reynolds_func)) then
      print *, 'Failed to get Reynolds function'
      return
    end if

    ! Test with density and viscosity
    args = PyTuple_New(0_c_size_t)
    kwargs = PyDict_New()
    
    call PyDict_SetItemString(kwargs, "V"//C_NULL_CHAR, PyFloat_FromDouble(2.5d0))
    call PyDict_SetItemString(kwargs, "D"//C_NULL_CHAR, PyFloat_FromDouble(0.25d0))
    call PyDict_SetItemString(kwargs, "rho"//C_NULL_CHAR, PyFloat_FromDouble(1.1613d0))
    call PyDict_SetItemString(kwargs, "mu"//C_NULL_CHAR, PyFloat_FromDouble(1.9d-5))
    
    result = PyObject_Call(reynolds_func, args, kwargs)
    if (c_associated(result)) then
      Re1 = PyFloat_AsDouble(result)
      print *, '✓ Re (with rho, mu): ', Re1
      
      ! Assert abs(Re1 - 38200.6579) < 0.1
      test_passed = abs(Re1 - 38200.6579d0) < 0.1d0
      if (test_passed) then
        print *, 'Assert passed: |Re1 - 38200.6579| < 0.1'
      else
        print *, 'Assert failed: |Re1 - 38200.6579| < 0.1'
      end if
    end if

    ! Test with kinematic viscosity
    args = PyTuple_New(0_c_size_t)
    kwargs = PyDict_New()
    
    call PyDict_SetItemString(kwargs, "V"//C_NULL_CHAR, PyFloat_FromDouble(2.5d0))
    call PyDict_SetItemString(kwargs, "D"//C_NULL_CHAR, PyFloat_FromDouble(0.25d0))
    call PyDict_SetItemString(kwargs, "nu"//C_NULL_CHAR, PyFloat_FromDouble(1.636d-5))
    
    result = PyObject_Call(reynolds_func, args, kwargs)
    if (c_associated(result)) then
      Re2 = PyFloat_AsDouble(result)
      print *, '✓ Re (with nu): ', Re2
      
      ! Assert abs(Re2 - 38202.934) < 0.1
      test_passed = abs(Re2 - 38202.934d0) < 0.1d0
      if (test_passed) then
        print *, 'Assert passed: |Re2 - 38202.934| < 0.1'
      else
        print *, 'Assert failed: |Re2 - 38202.934| < 0.1'
      end if
    end if

  end subroutine test_reynolds  
end module
  
program main
  use python_utils
  implicit none
  
  call init_python()
  call test_fluids()
  call test_tank()
  call test_reynolds()
  call benchmark_fluids()
  
  print *, 'All tests completed!'
end program main
