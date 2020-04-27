nohup python3 -u motion_detector.py &
echo $! > save_pid.txt
A="Starting motion detector with PID"
B=" $(cat save_pid.txt)"
echo $A$B