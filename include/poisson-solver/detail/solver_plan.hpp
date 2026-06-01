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
#include <span>

namespace poisson_solver {
// === Forward Declarations ===
template <std::floating_point T, std::size_t Dim>
    requires(Dim <= 3 && Dim >= 1)
class Solver;
}  // namespace poisson_solver

namespace poisson_solver::detail {

// === Structs ===
// Holds the behaviour and configuration
// of the solver that is dependant on BCs.
// Doesn't manage any memory.
template <std::floating_point T, std::size_t Dim>
struct SolverPlan {
public:
    using AxisStrategies =
        std::array<std::unique_ptr<const IAxisStrategy<T>>, Dim>;
    using SolveMethod = void (poisson_solver::Solver<T, Dim>::*)();

    AxisStrategies axis_strategies{};
    SolveMethod solve_method{};

    // Default is `FullLayout` which will be changed
    // if we realize that we don't have full BCs.
    VariantLayout fft_layout{FullLayout{}};

    explicit SolverPlan(BCs<Dim> bcs)
        : axis_strategies(GetAxisStrategies(bcs)) {
        bool all_bcs_identical = TryConstructFromFullBCs(bcs);
        if (all_bcs_identical) {
            return;
        }
        ConstructFromNonFullBCs(bcs);
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

    // Tries to construct the object in the case
    // all BCs are identical as this behavior is
    // shared among all dimensions.
    auto TryConstructFromFullBCs(BCs<Dim> bcs) -> bool {
        using std::ranges::all_of;
        using enum FFTTransformType;

        auto &layout = std::get<FullLayout>(fft_layout);

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

        auto &layout = std::get<FullLayout>(fft_layout);
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
        fft_layout = SplitLayout{};

        auto &layout = std::get<SplitLayout>(fft_layout);
        auto [bcx, bcy, bcz] = bcs;

        if (IsPeriodic(bcx) && IsPeriodic(bcy) && IsNeumannOrStag(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePPNs;
            layout.forward_type_1d = GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;

        } else if (IsPeriodic(bcx) && IsPeriodic(bcy) && !HasPeriodic(bcz) &&
                   HasDirichletOrStag(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePPD;
            layout.forward_type_1d = GetAxisFFTType(bcz);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 2;

        } else if (IsNeumannOrStag(bcx) && IsPeriodic(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolveNsPP;
            layout.forward_type_1d = GetAxisFFTType(bcx);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 0;

        } else if (IsPeriodic(bcx) &&
                   ((IsNeumann(bcy) && IsNeumann(bcz)) ||
                    (IsNeumannStag(bcy) && IsNeumannStag(bcz)))) {
            solve_method = &Solver<T, Dim>::SolvePNsNs;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = GetAxisFFTType(bcy);
            layout.isolated_dim_idx = 0;

        } else if (IsPeriodic(bcx) && IsNeumannOrStag(bcy) && IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolvePNsP;
            layout.forward_type_1d = GetAxisFFTType(bcy);
            layout.forward_type_2d = Complex;
            layout.isolated_dim_idx = 1;

        } else if (((IsDirichlet(bcx) && IsDirichlet(bcy)) ||
                    (IsDirichletStag(bcx) && IsDirichletStag(bcy))) &&
                   IsPeriodic(bcz)) {
            solve_method = &Solver<T, Dim>::SolveDDP;
            layout.forward_type_1d = Complex;
            layout.forward_type_2d = GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;

        } else if (none_of(bcs, IsPeriodic) && (bcx == bcy) && (bcx != bcz) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, Dim>::Solve2Real1Real;
            layout.forward_type_1d = GetAxisFFTType(bcz);
            layout.forward_type_2d = GetAxisFFTType(bcx);
            layout.isolated_dim_idx = 2;

        } else if (none_of(bcs, IsPeriodic) && (bcy == bcz) && (bcx != bcy) &&
                   any_of(bcs, HasDirichletOrStag)) {
            solve_method = &Solver<T, Dim>::Solve1Real2Real;
            layout.forward_type_1d = GetAxisFFTType(bcx);
            layout.forward_type_2d = GetAxisFFTType(bcy);
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
            // In 2D is is only valid to use full BCs.
            throw std::invalid_argument("Invalid combination of 2D BCs.");
        } else if constexpr (Dim == 3) {
            ConstructFromNonFullBCs3D(bcs);
        }
    }
};
}  // namespace poisson_solver::detail

#endif  // SOLVER_PLAN_HPP
