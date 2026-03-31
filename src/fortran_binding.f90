module PoisSolver
    use iso_c_binding
    implicit none

    type, bind(C) :: PoissonSolver
        type(c_ptr)       :: handle    = c_null_ptr
        integer(c_size_t) :: grid_size = 0
    end type PoissonSolver

    integer(c_int), parameter :: PoisSolver_Periodic       = 0
    integer(c_int), parameter :: PoisSolver_Dirichlet      = 1
    integer(c_int), parameter :: PoisSolver_Neumann        = 2
    integer(c_int), parameter :: PoisSolver_DirichletStag  = 3
    integer(c_int), parameter :: PoisSolver_NeumannStag    = 4

    integer(c_int), parameter :: PoisSolver_Spectral               = 0
    integer(c_int), parameter :: PoisSolver_FiniteDifference2      = 1
    integer(c_int), parameter :: PoisSolver_FiniteDifference4      = 2

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

    ! Runs the solver, writing the solution into phi.
    ! phi and rhs must each contain solver%grid_size elements.
    subroutine poisson_solver_execute(solver, phi, rhs)
        type(PoissonSolver), intent(inout) :: solver
        real(c_double),      intent(out)   :: phi(*)
        real(c_double),      intent(in)    :: rhs(*)

        call poisson_solver_execute_c(solver, phi, rhs)
    end subroutine poisson_solver_execute

    subroutine poisson_solver_finalize(solver)
        type(PoissonSolver), intent(inout) :: solver

        call poisson_solver_finalize_c(solver)
    end subroutine poisson_solver_finalize

end module PoisSolver
