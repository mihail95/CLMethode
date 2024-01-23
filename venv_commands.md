### Create a virtual environment (creates /.venv)
``python -m venv .venv``

### Activate the environment
``.venv/Scripts/Activate.ps1`` - in Powershell\
Afterwards all pip commands will be run exclusively in the environment

### Generate a requirements file (for collaboration)
``pip freeze > requirements.txt``

### Install an existing requirements file
``pip install -r requirements.txt``

### Exit the environment
``deactivate``