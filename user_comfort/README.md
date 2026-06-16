# user_comfort

Plots and stats for application/ removal times and survey ratings.

Python 3.13.2

## Layout
- `data/` — CSV inputs (`app_times.csv`, `rem_times.csv`, `ratings.csv`)
- `notebooks/` — analysis notebook(s) -- 01 for initial plots, 02 for figure formating  
- `src/kin_tools/` — reusable code (utils, plotting, stats)
- `out/` — saved figures and tables

#### LME stats: https://www.statsmodels.org/stable/mixed_linear.html

#### Data collected here in 'Sheet1': https://docs.google.com/spreadsheets/d/1FZ8wFQWQy3xjgIYyGS5vvDAiU_qLpCyN-fVS316duq8/edit?usp=sharing


## Project tree: 
kinematics/user_comfort/
├─ data/
│  ├─ app_times.csv
│  ├─ rem_times.csv
│  └─ ratings.csv
├─ notebooks/
│  └─ 01_all_plots_and_stats.ipynb       # all plots shown here
├─ out/                                  # figures get saved here
├─ src/kin_tools/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ palettes.py
│  ├─ io.py
│  ├─ plotting.py                       # 
│  └─ stats.py                          #
└─ requirements.txt  
