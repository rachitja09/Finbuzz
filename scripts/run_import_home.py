import runpy
import traceback

try:
    runpy.run_path('pages/01_Home.py', run_name='__main__')
    print('OK')
except Exception as e:
    traceback.print_exc()
    print('ERROR', e)
