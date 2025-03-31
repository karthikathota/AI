# Project Setup Guide( macOS )

### 1. Install Pyenv and Pyenv-Virtualenv

```
brew update
brew install pyenv pyenv-virtualenv
```

Then, add the following lines to your shell configuration file (~/.zshrc or ~/.bashrc):

```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then apply the changes

```
source ~/.zshrc  # or source ~/.bashrc
exec $SHELL
```

### 2. Install a Python Version

This project uses 3.13.2

```
pyenv install 3.13.2
```

Set it as local

```
pyenv local 3.13.2
```

### 3. Create and Activate a Virtual Environment

Create a virtual environment named project1env

```
pyenv virtualenv 3.13.2 project1env
```

```
pyenv activate project1env
```

### 4. Install Required Packages

```
pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
```

### 5. Configure Jupyter Notebook to Use the Virtual Environment

```
pip install jupyter
```

Add the virtual environment as a Jupyter kernel:

```
python -m ipykernel install --user --name=project1env --display-name "Python (project1env)"
```

Now, run the jupyter notebook

```
jupyter notebook
```
