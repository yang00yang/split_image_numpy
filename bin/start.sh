
 #python split_image.py [src_dir] [dst_dir] <label_name>(可选,label.txt)
 nohup python -u  data_generator/split_image.py $1 $2 $3 ./logs/split.log 2>&1 &
