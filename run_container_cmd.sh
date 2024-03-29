#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Invalid number of arguments. I need a script to run and the cuda device string"
	exit -1
fi

singularity exec --cleanenv --contain --nv --bind ~/fakehome/:/home/ncarlotti\
		--bind ~/ssl_multi_led/:/ssl_multi_led --network-args "portmap=5000:5000/tcp"\
		--bind /media/cyan/self_supervised/nicholas/:/ssl_multi_led/data\
		--pwd /ssl_multi_led \
		~/containers/ssl_multi_led.sif \
		bash $1 $2