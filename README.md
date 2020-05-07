### RetinaNet Object Detection

Last update: May 2020.

---

This repository implements the RetinaNet architecture for single-shot object detction as described in [1], built on top of a ResNet-50 backbone.

#### Preliminaries

The object detection problem is joint *classification* and bounding box *regression*.
<p align="center"><img alt="$$&#10;\mathcal{L}(x, y, b) = \mathcal{L}_\mathrm{cls}(x,y) + \mathcal{L}_\mathrm{reg}(x, b)1\{y \neq 0\}&#10;$$" src="svgs/34adc2400501204b3c2c2e8f9dd62759.svg" align="middle" width="300.76825844999996pt" height="17.031940199999998pt"/></p>
Here <img alt="$x$" src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg" align="middle" width="9.39498779999999pt" height="14.15524440000002pt"/> is an image, <img alt="$y$" src="svgs/deceeaf6940a8c7a5a02373728002b0f.svg" align="middle" width="8.649225749999989pt" height="14.15524440000002pt"/> is a label, and <img alt="$b$" src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg" align="middle" width="7.054796099999991pt" height="22.831056599999986pt"/> is a bounding box with four coordinates.

How do we do this? Given an image, we'll create a set of *anchor boxes* at varous (1) locations, (2) sizes, and (3) aspect ratios. In the left image below we show the anchor boxes for a fixed scale, at the bottom-left and top-right corners. In right image below we show the anchor boxes fixed at the bottom-left corner and default aspect ratio, across various sizes.

![](svgs/anchors.png)

Each anchor box is responsible for its own <img alt="$(|\mathcal{Y}|+1)$" src="svgs/0a2ce5966757ab7e0ab3ec674df93546.svg" align="middle" width="62.56623614999999pt" height="24.65753399999998pt"/>-way classification problem for the label, as well as its own 4-way regression problem for the bounding box delta. 

Below <img alt="$(k, i, j, a)$" src="svgs/58c1c536d3e7bea06ffee29623a9a58c.svg" align="middle" width="64.92799829999998pt" height="24.65753399999998pt"/> denotes the <img alt="$a$" src="svgs/44bc9d542a92714cac84e01cbbb7fd61.svg" align="middle" width="8.68915409999999pt" height="14.15524440000002pt"/>-th anchor box at the <img alt="$(i, j)$" src="svgs/e8873e227619b7a62ee7eb981ef1faea.svg" align="middle" width="33.46496009999999pt" height="24.65753399999998pt"/> coordinate, corresponding to the <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align="middle" width="9.075367949999992pt" height="22.831056599999986pt"/>-th feature map.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\mathcal{L}_\mathrm{cls}^{(k, i, j, a)}(x,y) &amp; = -\log p_\theta(y|x) &amp; p_\theta(y|x) &amp; = \mathrm{Bernoulli},y \in \{0,1\}^{|\mathcal{Y}|}\\&#10;\mathcal{L}_\mathrm{reg}^{(k, i, j, a)}(x, b) &amp; = -\log p_\theta(b|x) &amp; p_\theta(b|x) &amp; = \mathrm{Normal}, b\in\mathbb{R}^4&#10;\end{align*}&#10;$$" src="svgs/db0f61122093f846ffd051f7c022b316.svg" align="middle" width="458.07092924999995pt" height="51.3808218pt"/></p>
RetinaNet uses a *feature pyramid network* that takes advantage of the hierarchical layers of a convolutional neural network to naturally predict at the different scales of the different anchor boxes. That is, the largest anchor boxes correspond to higher  level feature maps and the smallest anchor boxes correspond to lower level feature maps. By default there are ~5 feature maps for a ResNet-50 backbone.

At prediction time we use non-maximum suppression to address redundancy in the overlapping anchor boxes.

**Data Representation**

Unfortunately we need to break abstractions between the model and data loader. 

In order to produce valid targets for the neural network (i.e. for the classification and regression problems above), the data loader needs to transform an image plus its set of bounding boxes into:
<p align="center"><img alt="$$&#10;(x, y, b) \mapsto (\mathrm{cls\ target}\in \{0,1\}^{K, M, N, A, |\mathcal{Y}|, }, \mathrm{reg\ target}\in\mathbb{R}^{K, M, N, A, 4})&#10;$$" src="svgs/ccadef5136a2e99bfeb2e332bac5d359.svg" align="middle" width="478.79836125pt" height="19.526994300000002pt"/></p>
Note that above <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" align="middle" width="15.13700594999999pt" height="22.465723500000017pt"/> is the number of feature maps in the FPN, <img alt="$M,N$" src="svgs/cc7b1282b94a56c76e3f57e8a45e821e.svg" align="middle" width="39.132343799999994pt" height="22.465723500000017pt"/> are the number of coordinates, and <img alt="$A$" src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.32879834999999pt" height="22.465723500000017pt"/> the number of anchor boxes per coordinate. 

We choose to do this transform in the `collate_fn` of the dataloader. The advantage of doing the target transformation in the loader instead of the model is that we can take advantage of PyTorch default dataloader multi-processing.

#### Focal Loss

We implement the suggested RetinaNet focal loss,
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\mathcal{L}_\mathrm{cls}^{(k, i, j, a)}(x,y) &amp; = -\alpha^{(y)}(1-p_\theta(y|x)^\gamma\log p_\theta(y|x)\\&#10;p_\theta(y|x) &amp; = \mathrm{Bernoulli},\quad y \in \{0,1\}^{|\mathcal{Y}|}&#10;\end{align*}&#10;$$" src="svgs/9b29556995e5a285ca15923dd2426c2f.svg" align="middle" width="332.23793459999996pt" height="49.1891268pt"/></p>


Here <img alt="$\alpha^{(y)} = \alpha 1\{y=1\} + (1-\alpha)1\{y=0\}$" src="svgs/45720684f88c5b29a5736e7ba7aee85d.svg" align="middle" width="259.8970242pt" height="29.190975000000005pt"/> is a scaling factor to address class imbalance in the training data (since there are far more negative labels than positive labels). Note that setting <img alt="$\alpha=0.5$" src="svgs/766b87ee4c5af866353ddcb065b55b2a.svg" align="middle" width="53.49877169999999pt" height="21.18721440000001pt"/> and <img alt="$\gamma=0$" src="svgs/7eaedc1b9d7a4b11f78f1c63edf34f3a.svg" align="middle" width="39.56070194999999pt" height="21.18721440000001pt"/> recovers original log-likelihood. All final linear layer logits are initialized to prevalence level <img alt="$\pi=0.01$" src="svgs/d5a42a05d1f159a68487732e83987f68.svg" align="middle" width="61.101570749999986pt" height="21.18721440000001pt"/>.

#### Datasets

Here we support the MSCOCO [3] and Pascal VOC [4] datasets. 

See the `out/` directory for examples from trained models.

#### References

[1] T. Lin, P. Goyal, R. Girshick, K. He, & P. Dollar, Focal Loss for Dense Object Detection. *2017 IEEE International Conference on Computer Vision (ICCV)* (2017), pp. 2999–3007. https://doi.org/10.1109/ICCV.2017.324.

[2] K. He, X. Zhang, S. Ren, & J. Sun, Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (2016), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90.

[3] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, & C. L. Zitnick, Microsoft COCO: Common Objects in Context. In D. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars,eds., *Computer Vision – ECCV 2014* (Cham: Springer International Publishing, 2014), pp. 740–755. https://doi.org/10.1007/978-3-319-10602-1_48.

[4] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, & A. Zisserman, The Pascal Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, **88** (2010) 303–338. https://doi.org/10.1007/s11263-009-0275-4.