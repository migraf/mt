## Master Thesis Analysis Package
Underlying code of my master thesis

### Installing the package
```
git clone https://github.com/migraf/mt.git
```

```
pip install -e analysis
```

The package can then be imported

```
from analysis import *
```

### Docker image with jupyter notebook example

Download the image

```
docker pull grafmic/hypothesis:notebook
```

Run the image with a forwarded port for jupyter notebook

```
docker run -p 8888:8888 grafmic/hypothesis:notebook
```
Open the displayed url in the browser to access the notebook server
