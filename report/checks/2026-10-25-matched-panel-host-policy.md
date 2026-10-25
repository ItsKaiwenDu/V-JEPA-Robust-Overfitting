# Host policy findings after the interrupted panel

October 25, 2026. The read-only probe reports effective KillUserProcesses=false,
a ten-second user-manager stop delay and Linger=no. The current user slice and
user service report no fixed memory cap; the current CPU-time limit and CPU
quota are unlimited. These observations support ordinary nohup SSH detachment.
They do not prove what terminated the earlier queue or guarantee persistence.

System/kernel journal access is restricted. No own-user coredump metadata was
found. The current user-slice memory peak is only 35,741,696 bytes, so its zero
OOM counters do not cover the earlier full-model job. A past global/cgroup OOM,
admin action or other termination remains unproved; the host did not reboot.

Both completed cases and the partial third are preserved. A separate operator
recovery kit is prepared, with no new GPU run or server-setting change. The
old time allowance is conservatively exhausted; additional time requires a
recorded explicit authorization. See matched-panel-recovery-preparation.
