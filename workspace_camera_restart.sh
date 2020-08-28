# specify the project directory
dir=/home/am/projects/opencv_examples

# get timestamp  of tracking file modification
# and compare it with the current timestamp
filemtime=`stat -c %Y $dir/doorcam.txt`
currtime=`date +%s`
diff=$(( (currtime - filemtime) ))
echo $diff
# check whether the copy of the programm is running
status=$(ps aux | grep doorcam.py | grep python3 | wc -l)
echo $status
# run oly if tracking file hasn't been modified withing last 60 seconds
if [[ $diff -ge 60 ]]; then
    #  kill existing process running
    if [[ $status -ne 0 ]]; then
	# get pid of the process
	pid=$(ps -aux | grep doorcam.py | grep python3 | awk {'print $2'})
	kill -9 $pid
    fi
    # format current timestamp for log file
    now=$(date +"%F %T,%3N")
    # add ERROR message to log file
    echo "$now - root - ERROR - restarted a programm from cron" >> $dir/doorcam.log
    # start the process
    bash $dir/workspace_camera.sh
fi
