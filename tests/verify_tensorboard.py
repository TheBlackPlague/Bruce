import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


run = EventAccumulator(sys.argv[1]).Reload()
loss = run.Scalars("train/loss")


assert len(loss) == int(sys.argv[2]), loss
assert loss[0].step == 210 and loss[0].value == 0.125
assert loss[0].wall_time > 0
assert not any(tag.startswith("progress/") for tag in run.Tags()["scalars"])
assert math.isclose(run.Scalars("performance/mpos_per_second")[0].value, 0.0005, rel_tol=1e-6)
assert math.isclose(run.Scalars("train/learning_rate")[0].value, 0.001, rel_tol=1e-6)
assert not any("validation" in tag for tag in run.Tags()["scalars"])
