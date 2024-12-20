## EUP<sup>3</sup>M: Evaluating uncertainty and predictive performance of probabilistic models
## <font color="#cc0066">Docker Information</font>

- [Docker](https://www.docker.com) provides a platform for running the experiments
  in a stable environment.
- It uses OS-level virtualisation to deliver software in packages called containers.
- A [DockerFile](./DockerFile) is provided to assemble the docker image.
  - In our case, it derives from the `python:3.12.6-bookworm` base
    image, so it sets up [Python](https://www.python.org/about/) on Debian 12.

If users would prefer not to use Docker, then please refer to the
[requirements](./requirements-windows-python3.8.txt) file for a list of Python dependencies.


## Useful Commands

- Building Docker image
  - `docker build - < DockerFile`
    - Docker output is generally directed to the stderr, so
      <font color="#cc0066">`2>&1 | tee docker-build-$(date +%Y%m%d_%H%M%S).log`</font> may be added
      to the above command to capture everything in the console and log file.
- Tagging Docker image
  - `docker tag <HASH> eup3m:python3.12.6-debian12`
- Running Docker image
  - `docker run -it -v ${REPOS_PARENT_DIR}:/home/experiments --rm eup3m:python3.12.6-debian12 /bin/bash`
    - `--interactive --tty` (or `-it` for short) takes users inside the container.
    - `--volume` (or `-v <src>:<dst>`) mounts the specified directory on the host
      `<src>` inside the container at the specified path `<dst>`. Assuming the
      .git repository is located at <font color="#cc0066">`${REPOS_PARENT_DIR}/eup3m`</font>, this
      will make the source tree available as `/home/experiments/eup3m` inside the container.
    - `--rm` ensures the container is automatically removed when it is terminated.
    - `/bin/bash` overwrites the default python command and runs a shell instead.
- Running a Python script inside container
  - For instance, `cd eup3m/code/`
  - `python -m <script_name>`


## Requirement

- Download and install the latest version of Docker (&ge; 20.10.24).
  This can be checked using the command `docker version`.
  - Testing was conducted using Docker 26.1.4 on a RedHat 7 x86_64 machine
    with 32 (Intel Xeon E5-2680 @ 2.7GHz) CPUs.
- If using a Windows host, ensure bash scripts have Unix line endings. To avoid
  >     root@14f711c2d3f3:/home/experiments/eup3m/code# ./run_experiments.sh
  >     bash: ./run_experiments.sh: cannot execute: required file not found
  - Checkout source with LF line endings. For instance, by editing `.gitattributes` with
    - `code/*.sh text eol=lf`
  - OR from a bash prompt running inside the container
    - `apt install dos2unix; dos2unix run_experiments.sh`
