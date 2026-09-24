# Container environment

Along with `poisson-solver`, we also provide a container environment in which you can run the solver.
It is designed specifically to run `gpu-elmm` with `poisson-solver` (see [motivation](../docs/motivation.md) for information about both) on a cluster setup.
Use [links](../docs/README.md#links) to navigate to the `gpu-elmm` repository if you need it.

## Precautions
Ensure that all commands in this guide are run from the same worker node as the one on which you will be executing the code.
Failing to do so could result in errors regarding mismatched display driver / cuda driver versions.
E.g., when using `SLURM`, prefix all commands by `srun -w some_worker`.
Setups with heterogeneus node configurations are currently not supported.

## Building the container
First, recreate the following directory structure by creating a new directory somewhere on your filesystem, _copying_ `Dockerfile.atmos` into that directory, creating the `mapped` directory inside of it and _moving_ `poisson-solver` into `mapped`.
```
my-newly-created-container-directory
├── Dockerfile.atmos
└── mapped
    └── poisson-solver
```
`mapped` will function as a read-write directory for the container (that is why `poisson-solver` is inside of it).

Now, inside your newly created directory, we will build the image, convert it into a read-only directory named `imgdir` and inject the NVIDIA driver.
**Do not forget that the commands need to be run on a worker node where your code will execute.**
On a `SLURM` managed cluster we can e.g. execute the following command `salloc -p <partition-name> --gpus 1 -w <node-name>` to allocate a worker node with a GPU and then, prefix all commands from the following block with `srun`.
Beware that individual steps can take tens of minutes (allocate your jobs with high enough time limit).
You can _significantly_ speed this process up by using more CPUs (add `-c <num-of-cpus>` to your `salloc`).
```bash
ch-image build -t atmos -f Dockerfile.atmos .
ch-convert -i ch-image -o dir atmos imgdir
ch-fromhost --nvidia imgdir
```
If you followed the `SLURM` specific steps (running `salloc`), you can now relinquish the allocation by running `exit`.

## How to run code inside the container
To demonstrate how to run code inside the container, we will compile `poisson-solver` and `gpu-elmm` and then run an example.

Clone `gpu-elmm`:
```bash
cd mapped/
git clone https://github.com/mntjams/gpu-elmm.git
cd ..
```

Now, we build `poisson-solver` and `gpu-elmm`.
Again, **ensure the following commands are executed on a worker node where you plan to run your code**.
On a `SLURM` managed cluster we can again do that via `salloc -p <partition-name> --gpus 1 -w <node-name>` and then prefixing the commands from the following block with `srun`.
```bash
ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/poisson-solver && unset CXX && unset CC && cmake --preset release && cmake --build build/"
ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/gpu-elmm/src && unset CC && ./make_release"
```
**Running `unset CXX` (and `unset CC`) may be required whenever you start the container and intend to build the code.**
Charliecloud inherits host environment variables and those can point to invalid paths inside the container.

These commands will build the local version of `poisson-solver` and `gpu-elmm`.
To build the distributed version of `poisson-solver`, follow the guidance in the [examples README](../examples/README.md).
Distributed `gpu-elmm` must be built using `./make_mpi_release` instead of `./make_release`.

## Running an Example
You can now try one of the `gpu-elmm` examples (once again, from a worker node):
```bash
ch-run -b mapped:/opt/build imgdir -- /bin/bash -c "cd /opt/build/gpu-elmm/examples/channel && ../../bin/gcc/release/ELMM"
```

Distributed build of `gpu-elmm` stores the executable in `../../bin/gcc/mpi-release/ELMM`!
To run the example distributed follow similar approach to that shown in [examples](../examples/) (`UCX_POSIX_...` env. var set and `srun --mpi=pmix`).

## UCX permission denied
It is possible that you encounter this error when running distributed code inside this container.
This error stems from the fact that sibling Charliecloud containers cannot use `ptrace` on each other.
It is harmless but you can silence it by setting `UCX_POSIX_USE_PROC_LINK=n` (we use this for every automated run on a cluster, see e.g. [tests CMake](../tests/residuum/CMakeLists.txt)).

## Acknowledgments
[This repository](https://github.com/Erunno/elmm-in-clouds) served as an inspiration for the early versions of our `Dockerfile.atmos` and this guide.
