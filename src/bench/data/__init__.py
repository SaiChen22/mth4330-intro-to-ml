"""Dataset loaders, one module per dataset.

Rule (approval): data-dependent preprocessing (scaling, one-hot encoding) is fit on the
training split only.
Guideline: include data only when its license and size permit; otherwise provide download
links or scripts and document every preprocessing step.

Loader interface: TO DECIDE (docs/open-questions.md #11).
"""
