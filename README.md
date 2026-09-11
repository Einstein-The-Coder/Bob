# Bob

Ok, so the first thing you must do is **change directories** to your current project that you downloaded or cloned.

Next, create a virtual environment with

```
python -m venv .venv
```

Next, activate your .venv.

```
source /workspaces/Bob/.venv/bin/Activate.ps1
```

Now, install the packages.

```
pip install -r requirements.txt
```

To activate the training, run:

```
python train_ai.py
```

Also, if you want to change the train, you can change the variables at the top in ``` train_ai.py ```.


Now, to run the main file, ``` app.py ```.

```
python app.py
```

Now, the AI will be running on port 8000 (or whatever you set it to) and can be accessed via localhost.

So for example: ```locahost:8000``` and you replace the 8000 with whatever port you changed it to.
