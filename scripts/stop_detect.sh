A="Stopping motion detector with PID"
B=" $(cat save_pid.txt)"
echo $A$B

kill -9 `cat save_pid.txt`
rm save_pid.txt