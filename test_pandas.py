# %%
import pandas as pd
from datetime import datetime


# %%
df = pd.DataFrame(columns=['DateTime', 'Plate', 'Direction', 'Source'])

df.head()
# %%
df = df.append({'DateTime':datetime.now(), 'Plate':'ABF7059', 'Direction':'ENTERING', 'Source':'camera'}, ignore_index=True)
df.head()
# %%
df.to_excel('data.xlsx')
# %%
