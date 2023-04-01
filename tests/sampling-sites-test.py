import pandas as pd
import datetime
import pandas.testing as pdt

test_input = pd.DataFrame(
                data=[[2.0, 0.0],
                      [4.0, 0.0]],
                index=[pd.to_datetime('2000-01-01 01:00'),
                       pd.to_datetime('2000-01-01 02:00')],
                columns=['A', 'B']
)
test_result = pd.DataFrame(
                 data=[[3.0, 0.0]],
                 index=[datetime.date(2000, 1, 1)],
                 columns=['A', 'B']
)
pdt.assert_frame_equal(daily_mean(test_input), test_result)
