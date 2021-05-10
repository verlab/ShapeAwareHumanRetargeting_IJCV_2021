#  Parse the normal estimations codes to Python
```
cmake .
swig -c++ -python compute_normals.i
make
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Alternatively, you can export the PYTHONPATH in the last line of ```~/.bashrc``` in order to avoid export PYTHONPATH multiple times. Example:

```
export PYTHONPATH=$PYTHONPATH:<path_to_module>
```
