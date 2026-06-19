module PoisSolver
    use iso_c_binding
    implicit none

    type, bind(C) :: PoissonSolver
        type(c_ptr)       :: handle    = c_null_ptr
        integer(c_size_t) :: grid_size = 0
        integer(c_size_t) :: dim = 0
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

    ! === Uniform Solvers ===
    ! == Doubles ==
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

        subroutine poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecute")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: phi(*)
            real(c_double),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_c

        subroutine poisson_solver_finalize_c(solver) &
                bind(C, name="PoissonSolverFinalize")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_c

    end interface

    ! == Floats ==
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

        subroutine poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverExecuteFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: phi(*)
            real(c_float),      intent(in)    :: rhs(*)
            integer(c_int),     intent(in)    :: ngs_phi(*)
            integer(c_int),     intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_execute_float_c

        subroutine poisson_solver_finalize_float_c(solver) &
                bind(C, name="PoissonSolverFinalizeFloat")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_finalize_float_c

    end interface

    ! === Nonuniform Solvers ===
    ! == Doubles ==
    interface

         subroutine poisson_solver_nonuniform_z_new_c(solver_out, ns, lens, &
                 z, z_u, bcs, approx) &
                bind(C, name="PoissonSolverNonuniformZNew")
             import :: c_int, c_double, PoissonSolver
             type(PoissonSolver), intent(out) :: solver_out
             integer(c_int),      intent(in)  :: ns(*)
             real(c_double),      intent(in)  :: lens(*)
             real(c_double), intent(in) :: z(*)
             real(c_double), intent(in) :: z_u(*)
             integer(c_int),      intent(in)  :: bcs(*)
             integer(c_int),      value       :: approx
        end subroutine poisson_solver_nonuniform_z_new_c

        subroutine poisson_solver_nonuniform_z_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecute")
            import :: c_int, c_double, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_double),      intent(out)   :: phi(*)
            real(c_double),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_c

        subroutine poisson_solver_nonuniform_z_finalize_c(solver) &
                bind(C, name="PoissonSolverNonuniformZFinalize")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_nonuniform_z_finalize_c

    end interface

    ! == Floats ==
    interface

         subroutine poisson_solver_nonuniform_z_new_float_c(solver_out, ns, lens, &
                 z, z_u, bcs, approx) &
                bind(C, name="PoissonSolverNonuniformZNewFloat")
             import :: c_int, c_float, PoissonSolver
             type(PoissonSolver), intent(out) :: solver_out
             integer(c_int),      intent(in)  :: ns(*)
             real(c_float),      intent(in)  :: lens(*)
             real(c_float), intent(in) :: z(*)
             real(c_float), intent(in) :: z_u(*)
             integer(c_int),      intent(in)  :: bcs(*)
             integer(c_int),      value       :: approx
        end subroutine poisson_solver_nonuniform_z_new_float_c

        subroutine poisson_solver_nonuniform_z_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs) &
                bind(C, name="PoissonSolverNonuniformZExecuteFloat")
            import :: c_int, c_float, PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
            real(c_float),      intent(out)   :: phi(*)
            real(c_float),      intent(in)    :: rhs(*)
            integer(c_int),      intent(in)    :: ngs_phi(*)
            integer(c_int),      intent(in)    :: ngs_rhs(*)
        end subroutine poisson_solver_nonuniform_z_execute_float_c

        subroutine poisson_solver_nonuniform_z_finalize_float_c(solver) &
                bind(C, name="PoissonSolverNonuniformZFinalizeFloat")
            import :: PoissonSolver
            type(PoissonSolver), intent(inout) :: solver
        end subroutine poisson_solver_nonuniform_z_finalize_float_c
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
    ! === Uniform Solvers ===
    ! == Doubles ==

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
        
        call poisson_solver_new_c(solver, int(dims, c_int), &
            int(ns, c_int), lens, &
            int(bcs, c_int), int(approx, c_int))
    end subroutine poisson_solver_new

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_1d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:)
        real(c_double),      intent(in)    :: rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_2d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :)
        real(c_double),      intent(in)    :: rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_3d(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :, :)
        real(c_double),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d

    subroutine poisson_solver_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_c(solver)
    end subroutine poisson_solver_finalize

    ! == Floats ==

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
        
        call poisson_solver_new_float_c(solver, int(dims, c_int), &
            int(ns, c_int), lens, &
            int(bcs, c_int), int(approx, c_int))
    end subroutine poisson_solver_new_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_1d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:)
        real(c_float),      intent(in)    :: rhs(:)
        integer(c_int) :: ngs_phi(1), ngs_rhs(1)

        ngs_phi = int((shape(phi) - [solver%nx]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_1d_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_2d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :)
        real(c_float),      intent(in)    :: rhs(:, :)
        integer(c_int) :: ngs_phi(2), ngs_rhs(2)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_2d_float

    ! Expects that for each dimension the number
    ! of ghost cells on both sides are equal.
    subroutine poisson_solver_execute_3d_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :, :)
        real(c_float),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_execute_3d_float

    subroutine poisson_solver_finalize_float(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_float_c(solver)
    end subroutine poisson_solver_finalize_float

    ! === Nonuniform Solvers ===
    ! == Doubles ==
    subroutine poisson_solver_nonuniform_z_new(solver, ns, lens,&
            z, z_u, bcs, approx)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: ns(3)
        real(c_double),      intent(in)  :: lens(2)
        real(c_double),      intent(in)  :: z(:)
        real(c_double),      intent(in)  :: z_u(:)
        integer,             intent(in)  :: bcs(2 * 3)
        integer,             intent(in)  :: approx
        
        call poisson_solver_nonuniform_z_new_c(solver, int(ns, c_int), lens, &
            z, z_u, int(bcs, c_int), int(approx, c_int))
    end subroutine poisson_solver_nonuniform_z_new

    subroutine poisson_solver_nonuniform_z_execute(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(:, :, :)
        real(c_double),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute

    subroutine poisson_solver_nonuniform_z_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_nonuniform_z_finalize_c(solver)
    end subroutine poisson_solver_nonuniform_z_finalize

    ! == Floats ==
    subroutine poisson_solver_nonuniform_z_new_float(solver, ns, lens,&
            z, z_u, bcs, approx)
        type(PoissonSolver), intent(out) :: solver
        integer,             intent(in)  :: ns(3)
        real(c_float),      intent(in)  :: lens(2)
        real(c_float),      intent(in)  :: z(:)
        real(c_float),      intent(in)  :: z_u(:)
        integer,             intent(in)  :: bcs(2 * 3)
        integer,             intent(in)  :: approx
        
        call poisson_solver_nonuniform_z_new_float_c(solver, int(ns, c_int), lens, &
            z, z_u, int(bcs, c_int), int(approx, c_int))
    end subroutine poisson_solver_nonuniform_z_new_float

    subroutine poisson_solver_nonuniform_z_execute_float(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_float),      intent(out)   :: phi(:, :, :)
        real(c_float),      intent(in)    :: rhs(:, :, :)
        integer(c_int) :: ngs_phi(3), ngs_rhs(3)

        ngs_phi = int((shape(phi) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)
        ngs_rhs = int((shape(rhs) - [solver%nx, solver%ny, solver%nz]) / 2, c_int)

        call poisson_solver_nonuniform_z_execute_float_c(solver, phi, rhs, ngs_phi, ngs_rhs)
    end subroutine poisson_solver_nonuniform_z_execute_float

    subroutine poisson_solver_nonuniform_z_finalize_float(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_nonuniform_z_finalize_float_c(solver)
    end subroutine poisson_solver_nonuniform_z_finalize_float

end module PoisSolver
