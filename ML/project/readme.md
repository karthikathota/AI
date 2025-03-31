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
