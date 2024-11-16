k = 0
import time
import sys
while True:
    k += 1
    print(time.ctime())
    sys.stdout.flush()
    if k == 1000:
        k = 0
    
    # time.sleep(1)
    time.sleep(60*60)
