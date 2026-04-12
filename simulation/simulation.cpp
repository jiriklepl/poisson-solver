#include "poisson-solver/solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// --- Constants ---
const int GRID_SIZE = 200;
const float DT = 0.1f;
const float SOURCE_STRENGTH = 1.0f;
const float JET_VELOCITY =
    20.0f;  // Positive (Up logic depends on array indexing)
const float lenx = GRID_SIZE;
const float leny = GRID_SIZE;
const int ITERATIONS = 1000;

// --- Helper: Save to Binary ---
void SaveFrame(const poisson_solver::Workspace2D<float> &data, int frame_idx) {
    std::stringstream ss;
    ss << "frames/frame_" << std::setw(4) << std::setfill('0') << frame_idx
       << ".bin";

    std::ofstream out(ss.str(), std::ios::binary);
    if (out.is_open()) {
        out.write(reinterpret_cast<const char *>(data.Ptr()),
                  data.Size() * sizeof(float));
        out.close();
    }
}

// --- Helper: Linear Interpolation (Lerp) ---
float Lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// --- Helper: Periodic Sampling with Bilinear Interpolation ---
float SampleBilinear(const poisson_solver::Workspace2D<float> &field,
                     float x,
                     float y,
                     int N) {
    // 1. Handle Periodic Wrapping
    // fmod can return negative values, so we fix that.
    x = std::fmod(x, static_cast<float>(N));
    if (x < 0)
        x += N;

    y = std::fmod(y, static_cast<float>(N));
    if (y < 0)
        y += N;

    // 2. Find integer coordinates
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = (x0 + 1) % N;  // Wrap index
    int y1 = (y0 + 1) % N;  // Wrap index

    // 3. Get weights
    float sx = x - x0;
    float sy = y - y0;

    // 4. Sample neighbors
    // Row-Major Indexing: idx = y * N + x
    // float v00 = field[y0 * N + x0];
    // float v10 = field[y0 * N + x1];
    // float v01 = field[y1 * N + x0];
    // float v11 = field[y1 * N + x1];
    float v00 = field(y0, x0);
    float v10 = field(y0, x1);
    float v01 = field(y1, x0);
    float v11 = field(y1, x1);

    // 5. Interpolate
    float v0 = Lerp(v00, v10, sx);
    float v1 = Lerp(v01, v11, sx);
    return Lerp(v0, v1, sy);
}

// --- Physics: Semi-Lagrangian Advection ---
void Advect(poisson_solver::Workspace2D<float> &field,
            const poisson_solver::Workspace2D<float> &vel_x,
            const poisson_solver::Workspace2D<float> &vel_y,
            poisson_solver::Workspace2D<float> &new_field,
            int N) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            // int idx = j * N + i;

            // Backtrace: Where was this particle DT ago?
            // Note: In image coordinates, usually +y is down.
            // If you want standard physics (+y up), handle signs carefully.
            // float x_prev = i - DT * u[idx];
            // float y_prev = j - DT * v[idx];
            float x_prev = i - DT * vel_x(j, i);
            float y_prev = j - DT * vel_y(j, i);

            // new_field[idx] = SampleBilinear(field, x_prev, y_prev, N);
            new_field(j, i) = SampleBilinear(field, x_prev, y_prev, N);
        }
    }
}

// --- Main Simulation Loop ---
int main() {
    std::filesystem::create_directory("frames");
    int N = GRID_SIZE;

    // 1. Allocate Memory
    poisson_solver::Workspace2D density(N, N, 0.0f);
    poisson_solver::Workspace2D<float> density_temp(N, N);
    // std::vector<float> density(size, 0.0f);
    // std::vector<float> density_temp(size);
    poisson_solver::Workspace2D vel_x(N, N, 0.0f);
    poisson_solver::Workspace2D vel_y(N, N, 0.0f);
    // std::vector<float> u(size, 0.0f);  // Velocity X
    // std::vector<float> v(size, 0.0f);  // Velocity Y
    poisson_solver::Workspace2D<float> vel_x_temp(N, N);
    poisson_solver::Workspace2D<float> vel_y_temp(N, N);
    // std::vector<float> u_temp(size);
    // std::vector<float> v_temp(size);

    poisson_solver::Workspace2D<float> divergence(N, N);
    poisson_solver::Workspace2D<float> pressure(N, N);
    // std::vector<float> divergence(size);
    // std::vector<float> pressure(size);

    // 2. Initialize Solver (Assuming your class interface)
    // Using Periodic Boundaries for both X and Y
    poisson_solver::Solver2D solver(
        lenx,
        leny,
        N,
        N,
        {{poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic},
         {poisson_solver::BoundaryCondition::kPeriodic,
          poisson_solver::BoundaryCondition::kPeriodic}},
        poisson_solver::FiniteDifference2Approximation<float>{});

    std::cout << "Starting Simulation..." << std::endl;

    // 3. Time Stepping Loop
    for (int frame = 0; frame < ITERATIONS; ++frame) {
        // --- A. Advection ---
        // Advect velocity through itself (Self-advection)
        Advect(vel_x, vel_x, vel_y, vel_x_temp, N);
        Advect(vel_y, vel_x, vel_y, vel_y_temp, N);
        vel_x = vel_x_temp;
        vel_y = vel_y_temp;

        // Advect density through velocity
        Advect(density, vel_x, vel_y, density_temp, N);
        density = density_temp;

        // --- B. Add Forces (Source) ---
        // Place source in bottom-center
        int cx = N / 2;
        int cy = 10;  // Bottom (low index)
        int r = 3;

        for (int j = cy - r; j <= cy + r; ++j) {
            for (int i = cx - r; i <= cx + r; ++i) {
                // Bounds check (just in case)
                if (i >= 0 && i < N && j >= 0 && j < N) {
                    // int idx = j * N + i;
                    density(j, i) = SOURCE_STRENGTH;  // Add smoke
                    vel_y(j, i) = JET_VELOCITY;       // Push Up
                }
            }
        }

        // --- C. Project (Pressure Solve) ---

        // 1. Calculate Divergence (Finite Difference)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                // Periodic Neighbors
                int w = (i - 1 + N) % N;
                int e = (i + 1) % N;
                int s = (j - 1 + N) % N;
                int n = (j + 1) % N;

                // int idx = j * N + i;

                // Central Difference: (Right - Left) / 2
                // float du_dx = (vel_x[j * N + e] - vel_x[j * N + w]) * 0.5f;
                // float dv_dy = (vel_y[n * N + i] - vel_y[s * N + i]) * 0.5f;
                float du_dx = (vel_x(j, e) - vel_x(j, w)) * 0.5f;
                float dv_dy = (vel_y(n, i) - vel_y(s, i)) * 0.5f;

                divergence(j, i) = du_dx + dv_dy;
            }
        }

        // 2. Call YOUR Solver
        // Important: Ensure your solver expects (Solution, RHS)
        solver.Execute(pressure, divergence);

        // 3. Subtract Gradient (u = u - grad(p))
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int w = (i - 1 + N) % N;
                int e = (i + 1) % N;
                int s = (j - 1 + N) % N;
                int n = (j + 1) % N;

                // int idx = j * N + i;

                // float dp_dx =
                //     (pressure[j * N + e] - pressure[j * N + w]) * 0.5f;
                // float dp_dy =
                //     (pressure[n * N + i] - pressure[s * N + i]) * 0.5f;
                float dp_dx = (pressure(j, e) - pressure(j, w)) * 0.5f;
                float dp_dy = (pressure(n, i) - pressure(s, i)) * 0.5f;

                // u[idx] -= dp_dx;
                // v[idx] -= dp_dy;
                vel_x(j, i) -= dp_dx;
                vel_y(j, i) -= dp_dy;
            }
        }

        // --- D. Output ---
        if (frame % 1 == 0) {  // Save every frame
            SaveFrame(density, frame);
            std::cout << "Saved Frame " << frame << std::endl;
        }
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
