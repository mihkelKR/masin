#! /bin/bash

for file in *
do
   name="3_${file}"
   cp $file /home/mihkel/masin/pildid/kolmv6neli/pics/$name
done