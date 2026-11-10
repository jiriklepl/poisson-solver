# Resident device integration

`device_api.h` and Fortran module `poisson_solver_device` provide the version1
local uniform double-precision FD2/FD4 interface. Both dimensions and precision
are deliberately explicit: dimension3 is supported; other requests return
`POISSON_SOLVER_UNSUPPORTED`. Legacy void-return APIs remain source-compatible,
but integrations that require failure recovery must use this status ABI.

Use `cmake --preset release-sm86`, then `cmake --build build-sm86 --parallel 2`
and `cmake --install build-sm86 --prefix <prefix>`. All source dependencies are
pinned in `dependencies.lock.json`; an offline VkFFT checkout can be supplied as
`-DFETCHCONTENT_SOURCE_DIR_VKFFT=<checkout>`. Configuration writes generated
editor data into the build tree only. Examples, documentation and legacy Slurm
tests are optional. The installed CMake package exports
`PoissonSolver::poisson-solver`; generated Fortran modules are installed under
`include/poisson-solver/fortran` and require the matching compiler ABI. The C++
virtual device-execution signature gained a stream parameter; rebuild C++
consumers when adopting this version. Legacy C symbol signatures are unchanged.

Reserve the configuration's `device_budget_bytes` in the caller's residency
budget **before** `poisson_solver_create_v1`. Creation enforces that cap on
solver buffers and every `cudaMalloc` in the pinned VkFFT implementation,
including temporary transforms and prime-size LUT/Bluestein buffers. Query exact
live and peak owned device-buffer bytes afterward. The peak is the observed
creation/lifetime high-water, including any temporary planning buffers; after
successful creation shrink the provisional claim to current persistent bytes
plus the reported execution-extra requirement (zero in v1). CUDA context, loaded code/module and driver metadata
are explicitly outside these buffer counts. Exact workspace size is not claimed
before FFT planning. Execution refuses any additional device-buffer allocation attempt; retain the
claim until successful destruction. Events are execution resources, not field
payload bytes.

Pass valid dense first-index-fastest device allocations and their exact padded
byte sizes, separate nonoverlapping RHS/output, and explicit ghost counts. The
solver writes the phi interior only. RHS and phi halos remain unchanged.
`void* stream` carries the caller's `cudaStream_t`; all packing, FFT and numerical
kernels use that stream. Success may return before completion. The caller must
keep fields alive until its stream completes. One provider owns private
workspaces, serializes host submissions and records a private event so later
calls on another stream wait before workspace reuse. Creation and destruction
must run on the same CUDA device; destruction waits its last completion only. After a failed enqueue or event-record
attempt it joins the supplied stream before retirement instead. Keep that stream
alive until successful destruction.

Every execution sets `started=0` before preflight and `started=1` before the first
enqueue attempt. An error after that transition poisons the provider. Never
replay original numerical work after a started failure. No C++ exception crosses
the new ABI. Diagnostic text is thread-local and retrieved through
`poisson_solver_last_error_v1`. Handles must be live values returned by creation;
the caller must not copy or reuse a destroyed handle. Caller retirement must be
serialized with its submissions.

Build CPU ABI tests with `-DPOISSON_SOLVER_BUILD_API_TESTS=ON`; these do not
initialize CUDA. Differential GPU checks additionally require
`POISSON_SOLVER_POISFFT_BUILD` pointing to an authenticated build of pinned
PoisFFT commit `256a2338254c08f611881b2df5e72bdfa73bc257` and
`POISSON_SOLVER_POISFFT_LIBS` listing its FFTW libraries. Record hashes of the six
reference objects, modules, source checkout and FFTW inputs in the execution
receipt. Run `ctest -R 'device-(api|fortran)-contract'` for CPU checks and
`ctest -R '^poisfft-differential$'` only with an actual GPU correctness lease.

The differential cases cover FD2/FD4 on even and odd grids, periodic and
Neumann-staggered null spaces, and exact channel boundary classes
`[Periodic,Periodic,Periodic,Periodic,NeumannStag,NeumannStag]`. Ghost widths are
phi2/RHS1. They compare the active solutions against unmodified PoisFFT, check
zero-mean gauge/null RHS behavior, preserve RHS and phi-halo bytes, and use a
dependent producer in a nonblocking stream. They establish operator and memory
compatibility on those cases, not whole-application equivalence or performance.
