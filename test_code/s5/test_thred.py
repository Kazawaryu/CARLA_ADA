import atexit
import subprocess

def run_next_program():
    # subprocess.run(['python', "./thread2.py"])
    # 执行在终端中输入的命令作为下一个程序，输入： bash thread2.sh 4 --extra_tag 6 --batch_size 16
    subprocess.run(['bash', "./thread2.sh", "4 --extra_tag 6 --batch_size 16"])
    

# 在命令行中输入执行：bash thread1.sh 4 --extra_tag 6 --batch_size 16
subprocess.run(['bash', "./thread2.sh", "2 --extra_tag 6 --batch_size 16"])
atexit.register(run_next_program)