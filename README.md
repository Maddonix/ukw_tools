# ukw_tools
Tools for the workgroup InExEn.

## Requirements
### Update requirements
Command: pipreqs . --force

## Commit
### Bumpversion
Update the packages version by entering "bumpversion major" (minor, patch)
Add all files to include in the .bumpversion.cfg file.
### Create new Build:
python3 -m build
### Upload to pypi
python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*