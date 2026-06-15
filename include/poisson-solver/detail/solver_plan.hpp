#ifndef SOLVER_PLAN_HPP
#define SOLVER_PLAN_HPP

#include "poisson-solver/boundary_conditions.hpp"
#include "poisson-solver/detail/axis_strategies.hpp"
#include "poisson-solver/detail/fft_policies.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>

namespace poisson_solver {
// === Forward Declarations ===
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class Solver;

template <std::floating_point T>
class NonuniformZSolver;
}  // namespace poisson_solver

namespace poisson_solver::detail {

// === Structs ===

// Holds the behaviour and configuration
// of the solver that is dependant on BCs.
// Doesn't manage any memory.
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
struct ISolverPlan {
public:
    using AxisStrategies =
        std::array<std::unique_ptr<const IAxisStrategy<T>>, Dim>;

    AxisStrategies axis_strategies{};
    VariantLayout fft_layout{};

protected:
    explicit ISolverPlan(BCs<Dim> bcs, VariantLayout default_layout)
        : axis_strategies(GetAxisStrategies(bcs)),
          fft_layout(default_layout) {}

    // This function should be called with
    // an already validated BC pair.
    // Caller is responsible for throwing in
    // case boundary conditions aren't valid.
    auto GetAxisFFTType(BCPair bcp) -> FFTTransformType {
        using enum FFTTransformType;

        if (IsPeriodic(bcp)) {
            return Complex;
        }
        if (IsDirichlet(bcp)) {
            return DST1;
        }
        if (IsDirichletStag(bcp)) {
            return DST2;
        }
        if (IsNeumann(bcp)) {
            return DCT1;
        }
        if (IsNeumannStag(bcp)) {
            return DCT2;
        }
        if (IsDirichletNeumann(bcp)) {
            return DST3;
        }
        if (IsNeumannDirichlet(bcp)) {
            return DCT3;
        }
        if (IsDirichletStagNeumannStag(bcp)) {
            return DST4;
        }
        if (IsNeumannStagDirichletStag(bcp)) {
            return DCT4;
        }
        assert(false && "Invalid BC pair.");
    }

private:
    static auto GetAxisStrategy(BCPair bcp)
        -> std::unique_ptr<IAxisStrategy<T>> {
        if (IsPeriodic(bcp)) {
            return std::make_unique<PeriodicStrategy<T>>();
        }
        if (IsDirichlet(bcp)) {
            return std::make_unique<DirichletStrategy<T>>();
        }
        if (IsNeumann(bcp)) {
            return std::make_unique<NeumannStrategy<T>>();
        }
        if (IsDirichletStag(bcp)) {
            return std::make_unique<DirichletStagStrategy<T>>();
        }
        if (IsNeumannStag(bcp)) {
            return std::make_unique<NeumannStagStrategy<T>>();
        }
        if (IsDirichletNeumann(bcp) || IsNeumannDirichlet(bcp) ||
            IsDirichletStagNeumannStag(bcp) ||
            IsNeumannStagDirichletStag(bcp)) {
            return std::make_unique<MixedStrategy<T>>();
        }
        throw std::invalid_argument("Invalid boundary condition combination.");
    }

    static auto GetAxisStrategies(BCs<Dim> bcs) -> AxisStrategies {
        AxisStrategies strats{};
        for (std::size_t d = 0; d < Dim; ++d) {
            strats[d] = GetAxisStrategy(bcs[d]);
        }
        return strats;
    }
};

template <std::floating_point T, std::size_t Dim>
struct SolverPlan : public ISolverPlan<T, Dim> {
public:
    using SolveMethod = void (poisson_solver::Solver<T, Dim>::*)();

    SolveMethod solve_method{};

    explicit SolverPlan(BCs<Dim> bcs)
        : ISolverPlan<T, Dim>(bcs, FullLayout{}) {
        bool all_bcs_identical = TryConstructFromFullBCs(bcs);
        if (all_bcs_identical) {
            return;
        }
        ConstructFromNonFullBCs(bcs);
    }

private:
    // Tries to construct the object in the case
    // all BCs are identical as this behavior is
    // shared among all dimensions.
    auto TryConstructFromFullBCs(BCs<Dim> bcs) -> bool {
        using std::ranges::all_of;
        using enum FFTTransformType;

        auto &layout = std::get<FullLayout>(this->fft_layout);

        if (all_of(bcs, IsPeriodic)) {
            solve_method = &Solver<T, Dim>::SolveFullPeriodic;
            layout.forward_type = Complex;
        } else if (all_of(bcs, IsDirichlet)) {
            solve_method = &Solver<T, Dim>::SolveFullDirichlet;
            layout.forward_type = DST1;
        } else if (all_of(bcs, IsDirichletStag)) {
            solve_method = &Solver<T, Dim>::SolveFullDirichlet;
            layout.forward_type = DST2;
        } else if (all_of(bcs, IsNeumann)) {
            solve_method = &Solver<T, Dim>::SolveFullNeumann;
            layout.forward_type = DCT1;
        } else if (all_of(bcs, IsNeumannStag)) {
            solve_method = &Solver<T, Dim>::SolveFullNeumann;
            layout.forward_type = DCT2;
        } else {
            return false;
        }
        return true;
    }

    void ConstructFromNonFullBCs1D(BCs<1> bcs) {
        using enum FFTTransformType;

        auto &layout = std::get<FullLayout>(this->fft_layout);
        BCPair bcp = bcs[0];

        if (IsDirichletNeumann(bcp)) {
            layout.forward_type = DST3;
        } else if (IsNeumannDirichlet(bcp)) {
            layout.forward_type = DCT3;
        } else if (IsDirichletStagNeumannStag(bcp)) {
            layout.forward_type = DST4;
        } else if (IsNeumannStagDirichletStag(bcp)) {
            layout.forward_type = DCT4;
        } else {
            throw std::invalid_argument("Invalid combination of 1D BCs.");
        }

        // In all non-throwing cases the solve method is full dirichlet.
        solve_method = &Solver<T, Dim>::SolveFullDirichlet;
    }

    void ConstructFromNonFullBCs3D(BCs<3> bcs) {
        using std::ranges::any_of;
        using std::ranges::none_of;
        using enum FFTTransformType;

        // Non-full 3D case => layout must be split.
        this->fft_layout = SplitLayout{};

        auto &layout = std::get<SplitLayout>(this->fft_layout);
        auto [bcx, bcy, bcz] = bcs;

        if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePPNs;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;

        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && !HasPeriodic(bcz) &&
                   HasDirichletOrStag(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePPD;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;

        } else if (IsNeumannOrStag(bcx) && IsPeriodic(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolveNsPP;
            layout.forward_type_1d = this->GetAxisFFTType(bcx);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 0;

        } else if (IsPeriodic(bcx) &&
                   ((IsNeumann(bcy) && IsNeumann(bcz)) ||
                    (IsNeumannStag(bcy) && IsNeumannStag(bcz)))) {
            solve_method = &Solver<T, Dim>::SolvePNsNs;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = this->GetAxisFFTType(bcy);
            layout.isolated_dim_idx = 0;

        } else if (IsPeriodic(bcx) && IsNeumannOrStag(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePNsP;
            layout.forward_type_1d = this->GetAxisFFTType(bcy);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 1;

        } else if (((IsDirichlet(bcx) && IsDirichlet(bcy)) ||
                    (IsDirichletStag(bcx) && IsDirichletStag(bcy))) &&
                   IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolveDDP;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;

        } else if (none_of(bcs, IsPeriodic) && (bcx == bcy) && (bcx != bcz) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, Dim>::Solve2Real1Real;
            layout.forward_type_1d = this->GetAxisFFTType(bcz);
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;

        } else if (none_of(bcs, IsPeriodic) && (bcy == bcz) && (bcx != bcy) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, Dim>::Solve1Real2Real;
            layout.forward_type_1d = this->GetAxisFFTType(bcx);
            layout.forward_type_2d = this->GetAxisFFTType(bcy);
            layout.isolated_dim_idx = 0;

        } else {
            throw std::invalid_argument("Invalid combination of 3D BCs.");
        }
    }

    // Fallback option if all BCs are not identical.
    void ConstructFromNonFullBCs(BCs<Dim> bcs) {
        if constexpr (Dim == 1) {
            ConstructFromNonFullBCs1D(bcs);
        } else if constexpr (Dim == 2) {
            // In 2D it is only valid to use full BCs.
            throw std::invalid_argument("Invalid combination of 2D BCs.");
        } else if constexpr (Dim == 3) {
            ConstructFromNonFullBCs3D(bcs);
        }
    }
};

/**
 * struct NonuniformZSolverPlan - Plan for a solver nonuniform in Z axis.
 *
 * @tparam T Floating point type.
 */
template <std::floating_point T>
struct NonuniformZSolverPlan : public ISolverPlan<T, 2> {
public:
    using SolveMethod = void (poisson_solver::NonuniformZSolver<T>::*)();

    SolveMethod solve_method{};

    /**
     * @note All three BCs are required to check their correctness for
     * nonuniform Z solver.
     */
    explicit NonuniformZSolverPlan(BCs<3> bcs)
        : ISolverPlan<T, 2>(BCs<2>{bcs[0], bcs[1]}, SplitLayout{}) {
        using std::ranges::all_of;
        using enum FFTTransformType;

        auto &layout = std::get<SplitLayout>(this->fft_layout);
        auto [bcx, bcy, bcz] = bcs;

        if (all_of(bcs, IsNeumann) || all_of(bcs, IsNeumannStag)) {
            solve_method = &NonuniformZSolver<T>::SolveFullNeumann;
            layout.forward_type_2d = this->GetAxisFFTType(bcx);
        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            solve_method = &NonuniformZSolver<T>::SolvePPNs;
            layout.forward_type_2d = Complex;
        } else {
            throw std::invalid_argument(
                "Invalid combination of nonuniform BCs.");
        }
    }
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_PLAN_HPP
