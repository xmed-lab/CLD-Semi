path=$1
cmds=${@:2}

nohup python -u $path $cmds >> logs/__nohup/$(date +%s).log 2>&1 &
# python -u $path $cmds