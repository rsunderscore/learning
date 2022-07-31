# getting started
download client from [here](https://docs.docker.com/get-started/) 

- if on windows you will need [wls](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) (windows subsystem for linux)

by default images will be auto-downloaded from hub.docker.com

you will need to login to get images
- the login via docker desktop didn't work for me
- CLI login command is `docker login --username <username>`
  
 start a docker container
 - `docker run <imagename>`
 - useful flags - some of these are being replaced by a more verbose syntax
    - `-t` start with an interactive terminal shell
    - `-p #:#` map host port to container port
    - `-d` detached mode (no terminal) aka headless
    - `-v <hostloc>:<contloc>` create a volume mapped to a host folder to persist data - called a 'bind mount'
      - alternative `docker volume` creates a named storage location within the DockerDesktop data folders (e.g. C:\ProgramData\DockerDesktop)
      - `docker volume inspect <volname>` gives json definition for the volume: driver, mptpt etc...
      - e.g. you can use a bind mount for htdocs to an apache httpd container to serve those files

## watch container logs with `docker logs -f <container-id>`

# csutomize the container image
  generally this is done with a file named dockerfile
  
  subsequently you can run docker build  e.g. `docker build <dir>` ; where `<dir>` is the directory/folder to search for the dockerfile to use for build
  - `-t <tagname>` : tag the image during build
  
# repos and push
once you have a customized image you can push it to docker hub so you can get it from elsewhere later
1. create teh repos on docker hub by logging in and clicking to create new repos
1. docker tag <imagename> <username>/<reposname>
  - repos name is typically the same as your imagename
1. `docker push <username>/<reposname>`
  - easy to modify the tag command i.e. remove <imagename> and replace tag with push

# Dockerfile
- must start with a base imge e.g. FROM ubuntu or FROM python or FROM httpd
- COPY is used to copy files from host to container image
- RUN is used to execute commands to further modify the image e.g. RUN apt-get -y update or RUN pip install pandas
- WORKDIR sets the location to work from in the filesystem
- 

# Python
- images is built on Debian - can use apt-get to modify (after running apt-get -y update
- apache installed into this image will use the split conf layout and require a2enmod
  - conf in /etc/apachde2
- sudo doesn't work on containers - the default account for terminal is root
