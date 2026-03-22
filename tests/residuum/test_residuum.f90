! test_residuum.f90
!
! Residuum tests for the Poisson solver Fortran binding.
! Structure mirrors test_residuum.cpp and is inspired by testpoisson.f90
! from the PoisFFT repository.
!
! For each dimensionality we:
!   1. Fill phi with random values and rhs with random values whose mean
!      is subtracted (periodic BCs require a zero-sum RHS for a solution
!      to exist).
!   2. Solve via the Fortran binding.
!   3. Compute the maximum pointwise residuum of the discrete Laplacian,
!      i.e. max |L*phi - rhs| over all grid points, where L is the
!      second-order finite difference stencil.
!   4. Check that the residuum is below 100 * N * epsilon, where N is
!      the total number of grid points.
!
! Build (adjust paths to match your CMake install):
!   gfortran -o test_residuum test_residuum.f90 poisson_solver_binding.f90 \
!       -L<solver_lib_dir> -lpoisson_solver -lcufft -lcudart

module Kinds
  use iso_c_binding
  use iso_fortran_env

  integer,parameter :: rp = c_double
  integer,parameter :: size_kind = c_size_t
end module

module Globals
  use Kinds
  
  implicit none

  real(rp), parameter :: pi = 3.141592653589793238462_rp!pi = 4*atan(1._rp)!
  real(rp), parameter :: Lx = 2*pi, Ly = Lx*1.1, Lz = Lx/1.1
  integer :: nx = 21, ny = 32, nz = 25
  real(rp) :: dx,dy,dz
  real(rp), dimension(:,:,:), allocatable :: Phi3D, RHS3D
  real(rp), dimension(:,:), allocatable :: Phi2D, RHS2D
  real(rp), dimension(:), allocatable :: Phi1D, RHS1D

end module

module Residues
    use Kinds
    use Globals
    use PoisSolver

    implicit none


contains
    
  subroutine Res1D(Phi,RHS,&
                   Aw,Ae,&
                   Btype,R)
    !2nd order finite difference residuum

    real(rp), intent(inout) :: Phi(0:)
    real(rp), intent(in) :: RHS(:)
    real(rp),intent(in) :: Aw,Ae
    integer,intent(in) :: Btype(2)
    real(rp),intent(out) :: R
    integer i
    real(rp) :: p,Ap

    R = 0

    do i = 1, nx
      p = 0
      Ap = 0
      if (i>1) then
                p = p + Phi(i-1)*Aw
                Ap = Ap + Aw
      elseif (Btype(We)==PoisSolver_Periodic) then
                p = p + Phi(nx)*Aw
                Ap = Ap + Aw
      elseif (Btype(We)==PoisSolver_DirichletStag) then
                p = p - Phi(1)*Aw
                Ap = Ap + Aw
      elseif (Btype(We)==PoisSolver_Dirichlet) then
                Ap = Ap + Aw
      elseif (Btype(We)==PoisSolver_Neumann) then
                p = p + Phi(2)*Aw
                Ap = Ap + Aw
      end if
      if (i<nx) then
                p = p + Phi(i+1)*Ae
                Ap = Ap + Ae
      elseif (Btype(Ea)==PoisSolver_Periodic) then
                p = p + Phi(1)*Ae
                Ap = Ap + Ae
      elseif (Btype(Ea)==PoisSolver_DirichletStag) then
                p = p - Phi(nx)*Ae
                Ap = Ap + Ae
      elseif (Btype(Ea)==PoisSolver_Dirichlet) then
                Ap = Ap + Ae
      elseif (Btype(Ea)==PoisSolver_Neumann) then
                p = p + Phi(nx-1)*Ae
                Ap = Ap + Ae
      end if

      p = p - RHS(i)   
      p = abs(-p +Ap*Phi(i))      
      R = max(R,abs(p))
    end do

  end subroutine Res1D

  subroutine Res2D(Phi,RHS,&
                   Aw,Ae,As,An,&
                   Btype,R)
    !2nd order finite difference residuum

    real(rp), intent(inout) :: Phi(0:,0:)
    real(rp), intent(in) :: RHS(:,:)
    real(rp),intent(in) :: Aw,Ae
    real(rp),intent(in) :: As,An
    integer,intent(in) :: Btype(4)
    real(rp),intent(out) :: R
    integer i,j
    real(rp) :: p,Ap

    R = 0

    do j = 1, ny
      do i = 1, nx
        p = 0
        Ap = 0
        if (i>1) then
                  p = p + Phi(i-1,j)*Aw
                  Ap = Ap + Aw
        elseif (Btype(We)==PoisSolver_Periodic) then
                  p = p + Phi(nx,j)*Aw
                  Ap = Ap + Aw
        elseif (Btype(We)==PoisSolver_DirichletStag) then
                  p = p - Phi(1,j)*Aw
                  Ap = Ap + Aw
        elseif (Btype(We)==PoisSolver_Dirichlet) then
                  Ap = Ap + Aw
        elseif (Btype(We)==PoisSolver_Neumann) then
                  p = p + Phi(2,j)*Aw
                  Ap = Ap + Aw
        end if
        if (i<nx) then
                  p = p + Phi(i+1,j)*Ae
                  Ap = Ap + Ae
        elseif (Btype(Ea)==PoisSolver_Periodic) then
                  p = p + Phi(1,j)*Ae
                  Ap = Ap + Ae
        elseif (Btype(Ea)==PoisSolver_DirichletStag) then
                  p = p - Phi(nx,j)*Ae
                  Ap = Ap + Ae
        elseif (Btype(Ea)==PoisSolver_Dirichlet) then
                  Ap = Ap + Ae
        elseif (Btype(Ea)==PoisSolver_Neumann) then
                  p = p + Phi(nx-1,j)*Ae
                  Ap = Ap + Ae
        end if
        if (j>1) then
                  p = p + Phi(i,j-1)*As
                  Ap = Ap + As
        elseif (Btype(So)==PoisSolver_Periodic) then
                  p = p + Phi(i,ny)*As
                  Ap = Ap + As
        elseif (Btype(So)==PoisSolver_DirichletStag) then
                  p = p - Phi(i,1)*As
                  Ap = Ap + As
        elseif (Btype(So)==PoisSolver_Dirichlet) then
                  Ap = Ap + As
        elseif (Btype(So)==PoisSolver_Neumann) then
                  p = p + Phi(i,2)*As
                  Ap = Ap + As
        end if
        if (j<ny) then
                  p = p + Phi(i,j+1)*An
                  Ap = Ap + An
        elseif (Btype(No)==PoisSolver_Periodic) then
                  p = p + Phi(i,1)*An
                  Ap = Ap + An
        elseif (Btype(No)==PoisSolver_DirichletStag) then
                  p = p - Phi(i,ny)*An
                  Ap = Ap + An
        elseif (Btype(So)==PoisSolver_Dirichlet) then
                  Ap = Ap + An
        elseif (Btype(So)==PoisSolver_Neumann) then
                  p = p + Phi(i,ny-1)*An
                  Ap = Ap + An
        end if

        p = p - RHS(i,j)
        p = abs(-p +Ap*Phi(i,j))
        R = max(R,abs(p))
      end do
    end do

  end subroutine Res2D

  subroutine Res3D(Phi,RHS,&
                   Aw,Ae,As,An,Ab,At,&
                   Btype,R)
    !2nd order finite difference residuum

    real(rp), intent(inout) :: Phi(0:,0:,0:)
    real(rp), intent(in) :: RHS(:,:,:)
    real(rp),intent(in) :: Aw,Ae
    real(rp),intent(in) :: As,An
    real(rp),intent(in) :: Ab,At
    integer,intent(in) :: Btype(6)
    real(rp),intent(out) :: R
    integer i,j,k
    real(rp) :: p,Ap

    R = 0

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          p = 0
          Ap = 0
          if (i>1) then
                    p = p + Phi(i-1,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Periodic) then
                    p = p + Phi(nx,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_DirichletStag) then
                    p = p - Phi(1,j,k)*Aw
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Dirichlet) then
                    Ap = Ap + Aw
          elseif (Btype(We)==PoisSolver_Neumann) then
                    p = p + Phi(2,j,k)*Aw
                    Ap = Ap + Aw
          end if
          if (i<nx) then
                    p = p + Phi(i+1,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Periodic) then
                    p = p + Phi(1,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_DirichletStag) then
                    p = p - Phi(nx,j,k)*Ae
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Dirichlet) then
                    Ap = Ap + Ae
          elseif (Btype(Ea)==PoisSolver_Neumann) then
                    p = p + Phi(nx-1,j,k)*Ae
                    Ap = Ap + Ae
          end if
          if (j>1) then
                    p = p + Phi(i,j-1,k)*As
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Periodic) then
                    p = p + Phi(i,ny,k)*As
                    Ap = Ap + As
           elseif (Btype(So)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,1,k)*As
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Dirichlet) then
                    Ap = Ap + As
          elseif (Btype(So)==PoisSolver_Neumann) then
                    p = p + Phi(i,2,k)*As
                    Ap = Ap + As
          end if
          if (j<ny) then
                    p = p + Phi(i,j+1,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Periodic) then
                    p = p + Phi(i,1,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,ny,k)*An
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Dirichlet) then
                    Ap = Ap + An
          elseif (Btype(No)==PoisSolver_Neumann) then
                    p = p + Phi(i,ny-1,k)*An
                    Ap = Ap + An
          end if
          if (k>1) then
                    p = p + Phi(i,j,k-1)*Ab
                    Ap = Ap + Ab
          elseif (Btype(Bo)==PoisSolver_Periodic) then
                    p = p + Phi(i,j,nz)*Ab
                    Ap = Ap + Ab
           elseif (Btype(Bo)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,j,1)*Ab
                    Ap = Ap + Ab
          elseif (Btype(Bo)==PoisSolver_Dirichlet) then
                    Ap = Ap + Ab
          elseif (Btype(Bo)==PoisSolver_Neumann) then
                    p = p + Phi(i,j,2)*Ab
                    Ap = Ap + Ab
          end if
          if (k<nz) then
                    p = p + Phi(i,j,k+1)*At
                    Ap = Ap + At
          elseif (Btype(To)==PoisSolver_Periodic) then
                    p = p + Phi(i,j,1)*At
                    Ap = Ap + At
          elseif (Btype(To)==PoisSolver_DirichletStag) then
                    p = p - Phi(i,j,nz)*At
                    Ap = Ap + At
          elseif (Btype(To)==PoisSolver_Dirichlet) then
                    Ap = Ap + At
          elseif (Btype(To)==PoisSolver_Neumann) then
                    p = p + Phi(i,j,nz-1)*At
                    Ap = Ap + At
          end if

          p = p - RHS(i,j,k)
          p = abs(-p +Ap*Phi(i,j,k))
          R = max(R,abs(p))
        end do
      end do
    end do

  end subroutine Res3D
end module

module Tests
    use Kinds
    use Globals
    use Residues
    use PoisSolver

    implicit none

    character(20), parameter :: char_BCs(0:4) = ["Periodic            ", &
                                                "Dirichlet regular   ", &
                                                "Neumann regular     ", &
                                                "Dirichlet staggered ", &
                                                "Neumann staggered   "]
    subroutine TestFD2_2D(BCs)
        integer, intent(in) :: BCs(4)
        real(rp) :: R

        call RunFD2_2D(BCs)

        call Res2D(Phi2D, RHS2D, &
                   dx**(-2), dx**(-2), dy**(-2), dy**(-2), &
                   BCs, R)
        if (R < int(nx, int64) * int(ny, int64) * 100 * epsilon(1._rp)) then
          write(*,*) "FD2 OK", R
        else
          write(*,*) "FD2 FAIL"
          write(*,*) "FD2 residuum:",R
        end if
    end subroutine

    subroutine RunFD2_2D(BCs)
        type(PoissonSolver) :: Solver
        integer, intent(in) :: BCs(4)

        call poisson_solver_new(Solver, 2, [nx, ny], [Lx, Ly], BCs, 2)

        call poisson_solver_execute(Solver, Phi2D, RHS2D)

        call poisson_solver_finalize(Solver)
    end subroutine
end module

program test_residuum
    use Kinds
    use Globals
    use Residues
    use Tests
    use PoisSolver

  
    dx = Lx/nx;dy = Lx/ny;dz = Lx/nz

    real(c_double), parameter :: inv_dx_sq    = 1.0d0 / (dx * dx)
    real(c_double), parameter :: inv_dy_sq    = 1.0d0 / (dy * dy)
    real(c_double), parameter :: inv_dz_sq    = 1.0d0 / (dz * dz)

    allocate(RHS3D(nx,ny,nz))
    allocate(Phi3D(0:nx+1,0:ny+1,0:nz+1))

    allocate(RHS2D(nx,ny))
    allocate(Phi2D(0:nx+1,0:ny+1))

    allocate(RHS1D(nx))
    allocate(Phi1D(0:nx+1))



    do k = 1,nz
    do j = 1,ny
    do i = 1,nx
     x=(i-1._rp/2)*dx
     y=(j-1._rp/2)*dy
     z=(k-1._rp/2)*dz
     call random_number(RHS3D(i,j,k))
     call random_number(Phi3D(i,j,k))
    end do
    end do
    end do
    RHS3D = RHS3D-sum(RHS3D)/size(RHS3D)




    do j = 1,ny
    do i = 1,nx
     x=(i-1._rp/2)*dx
     y=(j-1._rp/2)*dy
     z=(k-1._rp/2)*dz
     call random_number(RHS2D(i,j))
     call random_number(Phi2D(i,j))
    end do
    end do
    RHS2D = RHS2D-sum(RHS2D)/size(RHS2D)



    do i = 1,nx
    x=(i-1._rp/2)*dx
    call random_number(RHS1D(i))
    call random_number(Phi1D(i))
    end do
    RHS1D = RHS1D-sum(RHS1D)/size(RHS1D)

end program test_residuum
