import habit

if habit.is_available('autogluon'):
    # AutoGluon 装上了，可以用 AutoGluonTabularModel
    print("AutoGluon 可用")
else:
    err = habit.import_error('autogluon')
    print(f"AutoGluon 不可用：{err}")