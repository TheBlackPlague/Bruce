import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


run = EventAccumulator(sys.argv[1]).Reload()
loss = run.Scalars("train/loss")


assert len(loss) == 1, loss
assert loss[0].step == 210 and loss[0].value == 0.125
assert loss[0].wall_time > 0
assert run.Scalars("progress/positions")[0].value == 21000
assert run.Scalars("performance/positions_per_second")[0].value == 500
assert math.isclose(run.Scalars("train/learning_rate")[0].value, 0.001, rel_tol=1e-6)
assert not any("validation" in tag for tag in run.Tags()["scalars"])
