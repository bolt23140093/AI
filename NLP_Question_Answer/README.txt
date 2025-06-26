=====================
在Windows 執行
=====================

pip install jieba bs4 tqdm
python Question_Answer.py

下載 中文詞向量 cna.cbow.cwe_p.tar_g.512d.0.txt

https://drive.google.com/drive/folders/1JJzdlL6T-bPi_lZ1qCOpgFlS1dVf3Q37


=====================
在Raspberry Pi 執行
=====================
#先安裝pip3 (如果沒有pip3的話)
sudo apt-get install python3-pip
#安裝numpy
sudo apt-get install libatlas-base-dev
pip3 install numpy==1.19.5
#安裝jieba等套件
pip3 install jieba bs4 tqdm html5lib
#安裝telegram-bot套件
pip3 install python-telegram-bot
#執行
python3 Question_Answer.py
