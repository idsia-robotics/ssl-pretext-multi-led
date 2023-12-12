#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Invalid number of arguments. I need a script to run and the cuda device string"
	exit -1

singularity exec --cleanenv --contain --nv --bind ~/fakehome/:/home/ncarlotti\
		--bind ~/ssl_multi_led/:/ssl_multi_led --network-args "portmap=5000:5000/tcp"\
		--bind /media/cyan/self_supervised/nicholas/:/ssl_multi_led/data\
		~/containers/ssl_multi_led.sif \
		bash /ssl_multi_led/$1 $2
