import subprocess
import sys
import os

state_codes = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
               "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
               "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
               "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
               "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# state_codes = ["AL","AK", "AZ","CA"]
# state_codes = [ "CA","CO"]
print("Inversion")
count=0
for state_code in state_codes:
    # if count==2:
    #     break
    
    subprocess.run(["python3", "run_fed_avg_attacks.py", "--experiment", "0", "--name_state", state_code])
    
    # count+=1

# Loop to call hello.py 10 times with different arguments
# for i in range(1):
    # subprocess.run(["python", "run_fed_avg_attacks.py", "--experiment", "0", "--name_state", "CA"])