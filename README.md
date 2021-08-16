# MCPEventModeImageReconstruction

This repository contains Python based image reconstruction software that can reduce the raw data collected by multi-channel plate detector under event mode into standard radiograph.

# How to use

The easiest way to use it to start reducing the data is by cloning this repository into your data reduction env with:

```
git clone https://github.com/KedoKudo/MCPEventModeImageReconstruction
```

create a virtual environment if possible, then install the dependencies with

```
pip install -r requirement.txt
```

followed by an local install with

```
pip install -e .
```

Now you can directly call the script to start the reduction

```
mcp3_reducer RAW_MCP_DATA.h5
```

For additional control options, use 

```
mcp3_reducer -h
```

# How to contribute

If you would like to contribute, please fork this repo to your own account.
Install the dev environment dependencies with

```
pip install -r requirement_dev.txt
```

Once you finished implementing the new features or bug fixes, you can submit a pull request and the maintainer will review and merge it once it is ready.

> NOTE: `pre-commit` is used to guard the quality of the source code, please do not disable it during your local development.
