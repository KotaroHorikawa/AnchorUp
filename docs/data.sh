# ---------- Cityscapes ----------
# https://github.com/cemsaz/city-scapes-script
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=XXXX&password=XXX&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip


# ---------- VOC ----------
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# generate folder VOCdevkit/VOC2012
tar -xvf VOCtrainval_11-May-2012.tar

# ---------- DAVIS ----------
mkdir DAVIS
cd DAVIS
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
mv DAVIS/* ./
rmdir DAVIS

# The paper profile reads ImageNet train/ and val/ under ${dataroot}/imagenet/images/.
# Download them from https://image-net.org/.
