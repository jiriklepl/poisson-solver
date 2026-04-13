module PoisSolver
    use iso_c_binding
    implicit none

    type, bind(C) :: PoissonSolver
        type(c_ptr)       :: handle    = c_null_ptr
        integer(c_size_t) :: grid_size = 0
        ! These correspond to C dimensions (row-major)
        ! In Fortran the size of first dimension is nz, then ny and last dim is nx
        integer(c_size_t) :: nx = 1, ny = 1, nz = 1
    end type PoissonSolver

    integer(c_int), parameter :: PoisSolver_Periodic       = 0
    integer(c_int), parameter :: PoisSolver_Dirichlet      = 1
    integer(c_int), parameter :: PoisSolver_Neumann        = 2
    integer(c_int), parameter :: PoisSolver_DirichletStag  = 3
    integer(c_int), parameter :: PoisSolver_NeumannStag    = 4

    integer(c_int), parameter :: PoisSolver_Spectral               = 0
    integer(c_int), parameter :: PoisSolver_FiniteDifference2      = 2
    integer(c_int), parameter :: PoisSolver_FiniteDifference4      = 4

    ! Doubles
    interface

        subroutine poisson_solver_new_c(solver_out, dims, ns, lens, bcs, approx) &
                bind(C, name="PoissonSolverNew")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(out) :: solver_out
            integer(c_int),      value       :: dims
            integer(c_int),      value       :: approx
            integer(c_int),      intent(in)  :: ns(*)
            real(c_double),      intent(in)  :: lens(*)
            integer(c_int),      intent(in)  :: bcs(*)
        end subroutine poisson_solver_new_c

        subroutine poisson_solver_execute_c(solver, phi, rhs) &
                bind(C, name="PoissonSolverExecute")
            import :: c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: phi(*)
            real(c_double),      intent(in)    :: rhs(*)
        end subroutine poisson_solver_execute_c

        subroutine poisson_solver_finalize_c(solver) &
                bind(C, name="PoissonSolverFinalize")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_c

    end interface

    ! Floats
    interface

        subroutine poisson_solver_new_float_c(solver_out, dims, ns, lens, bcs, approx) &
                bind(C, name="PoissonSolverNewFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(out) :: solver_out
            integer(c_int),      value       :: dims
            integer(c_int),      value       :: approx
            integer(c_int),      intent(in)  :: ns(*)
            real(c_float),       intent(in)  :: lens(*)
            integer(c_int),      intent(in)  :: bcs(*)
        end subroutine poisson_solver_new_float_c

        subroutine poisson_solver_execute_float_c(solver, phi, rhs) &
                bind(C, name="PoissonSolverExecuteFloat")
            import :: c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: phi(*)
            real(c_float),      intent(in)    :: rhs(*)
        end subroutine poisson_solver_execute_float_c

        subroutine poisson_solver_finalize_float_c(solver) &
                bind(C, name="PoissonSolverFinalizeFloat")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_float_c

    end interface

    interface poisson_solver_execute
        module procedure poisson_solver_execute_1d
        module procedure poisson_solver_execute_2d
        module procedure poisson_solver_execute_3d
    end interface

    interface poisson_solver_execute_float
        module procedure poisson_solver_execute_1d_float
        module procedure poisson_solver_execute_2d_float
        module procedure poisson_solver_execute_3d_float
    end interface


contains
    ! Creates a new solver, writing the result into `solver`.
    !
    ! dims        : number of spatial dimensions (1, 2, or 3)
    ! ns(dims)    : grid point count per axis
    ! lens(dims)  : domain length per axis
    ! bcs(2*dims) : boundary conditions, laid out as
    !               [west, east, south, north, bottom, top]
    ! approx      : one of PoissonSolver_Spectral,
    !                      PoissonSolver_FiniteDifference2, 
    !                      PoissonSolver_FiniteDifference4
    subroutine poisson_solver_new(solver, dims, ns, lens, bcs, approx)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: dims
        integer,             intent(in)  :: ns(dims)
        real(c_double),      intent(in)  :: lens(dims)
        integer,             intent(in)  :: bcs(2 * dims)
        integer,             intent(in)  :: approx
        integer, parameter :: idxs(6) = [2,1,4,3,6,5]
        
        call poisson_solver_new_c(solver, int(dims, c_int), &
            int(ns(dims:1:-1), c_int), lens(dims:1:-1), &
            int(bcs(idxs(2 * dims:1:-1)), c_int), int(approx, c_int))
    end subroutine poisson_solver_new

    subroutine poisson_solver_execute_1d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:)
        real(c_double),      intent(in)    :: rhs(:)
        integer :: ngPhi(1)
        integer :: ngRHS(1)

        ngPhi = (shape(phi) - [solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%nx]) / 2

        call poisson_solver_execute_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%nx - 1))
    end subroutine poisson_solver_execute_1d

    subroutine poisson_solver_execute_2d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :)
        real(c_double),      intent(in)    :: rhs(:, :)
        integer :: ngPhi(2)
        integer :: ngRHS(2)

        ngPhi = (shape(phi) - [solver%ny, solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%ny, solver%nx]) / 2

        call poisson_solver_execute_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%ny - 1, &
                lbound(Phi, 2) + ngPhi(2) : lbound(Phi, 2) + ngPhi(2) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%ny - 1, &
                lbound(RHS, 2) + ngRHS(2) : lbound(RHS, 2) + ngRHS(2) + solver%nx - 1))
    end subroutine poisson_solver_execute_2d

    subroutine poisson_solver_execute_3d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :, :)
        real(c_double),      intent(in)    :: rhs(:, :, :)
        integer :: ngPhi(3)
        integer :: ngRHS(3)

        ngPhi = (shape(phi) - [solver%nz, solver%ny, solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%nz, solver%ny, solver%nx]) / 2

        call poisson_solver_execute_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%nz - 1, &
                lbound(Phi, 2) + ngPhi(2) : lbound(Phi, 2) + ngPhi(2) + solver%ny - 1, &
                lbound(Phi, 3) + ngPhi(3) : lbound(Phi, 3) + ngPhi(3) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%nz - 1, &
                lbound(RHS, 2) + ngRHS(2) : lbound(RHS, 2) + ngRHS(2) + solver%ny - 1, &
                lbound(RHS, 3) + ngRHS(3) : lbound(RHS, 3) + ngRHS(3) + solver%nx - 1))
    end subroutine poisson_solver_execute_3d

    subroutine poisson_solver_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_c(solver)
    end subroutine poisson_solver_finalize

    ! Creates a new solver, writing the result into `solver`.
    !
    ! dims        : number of spatial dimensions (1, 2, or 3)
    ! ns(dims)    : grid point count per axis
    ! lens(dims)  : domain length per axis
    ! bcs(2*dims) : boundary conditions, laid out as
    !               [west, east, south, north, bottom, top]
    ! approx      : one of PoissonSolver_Spectral,
    !                      PoissonSolver_FiniteDifference2, 
    !                      PoissonSolver_FiniteDifference4
    subroutine poisson_solver_new_float(solver, dims, ns, lens, bcs, approx)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: dims
        integer,             intent(in)  :: ns(dims)
        real(c_float),      intent(in)  :: lens(dims)
        integer,             intent(in)  :: bcs(2 * dims)
        integer,             intent(in)  :: approx
        integer, parameter :: idxs(6) = [2,1,4,3,6,5]
        
        call poisson_solver_new_float_c(solver, int(dims, c_int), &
            int(ns(dims:1:-1), c_int), lens(dims:1:-1), &
            int(bcs(idxs(2 * dims:1:-1)), c_int), int(approx, c_int))
    end subroutine poisson_solver_new_float

    subroutine poisson_solver_execute_1d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:)
        real(c_float),      intent(in)    :: rhs(:)
        integer :: ngPhi(1)
        integer :: ngRHS(1)

        ngPhi = (shape(phi) - [solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%nx]) / 2

        call poisson_solver_execute_float_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%nx - 1))
    end subroutine poisson_solver_execute_1d_float

    subroutine poisson_solver_execute_2d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :)
        real(c_float),      intent(in)    :: rhs(:, :)
        integer :: ngPhi(2)
        integer :: ngRHS(2)

        ngPhi = (shape(phi) - [solver%ny, solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%ny, solver%nx]) / 2

        call poisson_solver_execute_float_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%ny - 1, &
                lbound(Phi, 2) + ngPhi(2) : lbound(Phi, 2) + ngPhi(2) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%ny - 1, &
                lbound(RHS, 2) + ngRHS(2) : lbound(RHS, 2) + ngRHS(2) + solver%nx - 1))
    end subroutine poisson_solver_execute_2d_float

    subroutine poisson_solver_execute_3d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :, :)
        real(c_float),      intent(in)    :: rhs(:, :, :)
        integer :: ngPhi(3)
        integer :: ngRHS(3)

        ngPhi = (shape(phi) - [solver%nz, solver%ny, solver%nx]) / 2
        ngRHS = (shape(rhs) - [solver%nz, solver%ny, solver%nx]) / 2

        call poisson_solver_execute_float_c(solver, &
            phi(lbound(Phi, 1) + ngPhi(1) : lbound(Phi, 1) + ngPhi(1) + solver%nz - 1, &
                lbound(Phi, 2) + ngPhi(2) : lbound(Phi, 2) + ngPhi(2) + solver%ny - 1, &
                lbound(Phi, 3) + ngPhi(3) : lbound(Phi, 3) + ngPhi(3) + solver%nx - 1), &
            rhs(lbound(RHS, 1) + ngRHS(1) : lbound(RHS, 1) + ngRHS(1) + solver%nz - 1, &
                lbound(RHS, 2) + ngRHS(2) : lbound(RHS, 2) + ngRHS(2) + solver%ny - 1, &
                lbound(RHS, 3) + ngRHS(3) : lbound(RHS, 3) + ngRHS(3) + solver%nx - 1))
    end subroutine poisson_solver_execute_3d_float

    subroutine poisson_solver_finalize_float(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_float_c(solver)
    end subroutine poisson_solver_finalize_float

end module PoisSolver
