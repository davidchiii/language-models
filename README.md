# Toxic Media - language-models
A introduction to natural language processing and sentimental analysis through the scope of analyzing negative online behaviors.


# Requirements

## WSL 2
For Windows users, first install WSL2. Install link [here](https://learn.microsoft.com/en-us/windows/wsl/install)

### Powershell
> `wsl --install`

Containers are lightweight, abstractions at the application layer that packages code and depencies together. Sharing the same operating system kernel, containers take up less space and run as isolated namespaces.

## Docker
This project uses [Docker](https://docs.docker.com/engine/install). 

[Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) is an alternative for Windows users that provides a GUI alternative to the WSL subsystem. Understand the [tradeoffs](https://www.docker.com/blog/guest-blog-deciding-between-docker-desktop-and-a-diy-solution/) for using Docker Desktop.

### Update repository

```
sudo apt-get update
sudo apt-get upgrade
```

### Install Docker packages

```
sudo apt install docker-ce docker-ce-cli containerd.io
```

### Install VSCode 

Located [here](https://code.visualstudio.com/download).

Install DevContainers Extension

