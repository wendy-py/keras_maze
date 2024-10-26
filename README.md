### Motivation
All credits to medium insights.

I implemented using keras(pytorch backend) instead, it works well, improved on PEP8 warning and decided to share.

maze_game features:
* recreate env of game for computer to observe 
* just run! (see computer learned paths in pygame)
* save all models and best results

### Install Guide
copy and run (assume linux with python3):
```
python3 -m venv test
cd test
source bin/activate
git clone https://github.com/wendy-py/keras-maze.git
cd keras_maze.py
pip install keras-core pytorch pandas pygame tqdm
python maze_game.py
```
