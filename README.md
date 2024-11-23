# Pokemon Trainer!

## Run
First, set up a virtual environment and run it
```
python3 -m venv venv
source venv/bin/activate
```

Now, please do a little install with 
```
pip install -r requirements.txt
```

Run the `trainer.py` with 
```
python3 -m training.trainer
```

This outputs something like 
```
INFO:root:Trainer started on 10.36.159.40:10134
```

Paste the IP and port to `trainer_address` in `pokemon.py`
```
trainer_address = 10.36.159.40:10134
```

Now, run your instances of `pokemon.py` with 
```
python3 -m training.pokemon --port xxxx
```

