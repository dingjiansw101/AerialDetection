The code is useful for <a href="http://captain.whu.edu.cn/DOTAweb/">DOTA<a> or
<a href="http://captain.whu.edu.cn/ODAI/">ODAI<a>. The code provide the following function
<ul>
    <li>
        Load and image, and show the bounding box on it.
    </li>
    <li>
        Evaluate the result.
    </li>
    <li>
        Split and merge the picture and label.
    </li>
</ul>

### What is DOTA?
<p>
Dota is a large-scale dataset for object detection in aerial images. 
It can be used to develop and evaluate object detectors in aerial images. 
We will continue to update DOTA, to grow in size and scope and to reflect evolving real-world conditions.
Different from general object detectin dataset. Each instance of DOTA is labeled by an arbitrary (8 d.o.f.) quadrilateral.
For the detail of <strong style="color:blue"> DOTA-v1.0</strong>, you can refer to our 
<a href="https://arxiv.org/abs/1711.10398">paper</a>.
</p>

### What is ODAI?
<p>
ODAI is a contest of object detetion in aerial images on <a href"http://www.icpr2018.org/">ICPR'2018</a>. It is based on <a href="http://captain.whu.edu.cn/DOTAweb/">DOTA-v1<a>. The contest is ongoing now. 
</p>

### Installation
1. install swig
```
    sudo apt-get install swig
```
2. create the c++ extension for python
```
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

### Usage
1. For read and visualize data, you can use DOTA.py
2. For evaluation the result, you can refer to the "dota_evaluation_task1.py" and "dota_evaluation_task2.py"
3. For split the large image, you can refer to the "ImgSplit"
4. For merge the results detected on the patches, you can refer to the ResultMerge.py

An example is shown in the demo.
The subdirectory of "basepath"(which is used in "DOTA.py", "ImgSplit.py") is in the structure of
```
.
├── images
└── labelTxt
```

