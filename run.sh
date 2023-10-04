nohup deepspeed --include=localhost:0 --master_port 25000 app.py 7b --deepspeed 2>&1 &
nohup deepspeed --include=localhost:1 --master_port 26001 app.py 3b --deepspeed 2>&1 &
