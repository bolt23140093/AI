#!/usr/bin/env python
# coding: utf-8

# # 使用說明:
# 
#     1.) 安裝 jieba，在指令行輸入以下指令 (也許會需要sudo)：
#         pip install jieba 
#         pip install python-telegram-bot
#
#     2.) 下載詞向量   cna.cbow.cwe_p.tar_g.512d.0.txt
#        https://mega.nz/#!5LwDjZia!f77y-eWm90H3akg8mD9CqhOZ89NihirRKN4IT1SJ01Q
#        
# 


import numpy as np 
from nlp import *

Answer_list = [
  "在2樓使用",
  "本飯店未提供嬰兒床",
  "可到櫃台租借。費用後付",
  "無法保管貴重物品或容易損壞的行李，敬請見諒。", 
  "所有客房均可免費使用有線LAN、無線LAN來連接網路。",
  "入住為15:00以後；退房為11:00以前。",
  "您可以用信用卡付款",
  "在2樓可使用早餐",
  "飯店設有西餐廳和中式餐廳",
  "很抱歉，請避免和寵物同行住宿。但導盲犬、肢體輔助犬、導聾犬可以同行入住，如有需要，請與櫃台聯絡。",
  "客房費用請在入住時一次付清",
  "有的。健身房在飯店的10樓",
  "有的。游泳池在飯店的地下室1F",
  "有的。訂房時，請您預訂禁菸房",
  "想要像香港一樣被統的話，可以直接到對岸"]


# 開啟詞向量檔案
dim,word_vecs=load_WordVector()
word_feature=set_word_vector(word_vecs,dim)
avg_ans_emb=[]

for idx,ans in enumerate(Answer_list):
	avg_ans_emb.append(word_feature(ans))
    
def predict(question,answers,debug=False):
    unknown = "我不懂您在說什麼。"
    avg_dlg_emb=word_feature(question)

    if (debug):
        print(clean_text(question))

    max_idx = -1
    max_sim = -10
    # 在六個回答中，每個答句都取詞向量平均作為向量表示
    # 我們選出與dialogue句子向量表示cosine similarity最高的短句
    for idx in range(len(answers)):
        sim = cosine_similarity(avg_dlg_emb,avg_ans_emb[idx])
        if (debug):
            print("Ans #%d:%s:" %  (idx,clean_text(answers)))
            print("Similarity #%d: %f" % (idx, sim))
        if sim > max_sim:
            max_idx = idx
            max_sim = sim

    if max_sim < 0.4:
        return unknown
    else:
        return answers[max_idx]




from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
import logging
TOKEN = "YOUR KEY"



def commands(update, context):

    
    print(update.message.text)
    if update.message.text == '/hello'.lower() or update.message.text == 'hello'.lower() or update.message.text == '/start'    or update.message.text == 'start':
        update.message.reply_text(
                '您好 {} ， 我可以怎麼幫您 ？'.format(update.message.from_user.first_name))
    else:
        reply_msg = predict(update.message.text,Answer_list)
        update.message.reply_text(reply_msg)
        print(reply_msg)
        
                
if __name__ == '__main__':
    
    

    #---------start TelegramBot ------------

    #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    updater = Updater(TOKEN, use_context=True)
    commands_handler = MessageHandler(Filters.text, commands)
    updater.dispatcher.add_handler(commands_handler)
    
    #method1: use webhook (https URL)
    #PORT 3080
    #./ngrok http 3080  ==> copy https URL string to HTTPS_URL
   
    PORT = int(os.environ.get('PORT', '3080'))
    HTTPS_URL="https://5062-220-132-124-155.ngrok.io/"  
    # add handlers
    updater.start_webhook(listen="0.0.0.0", port=PORT,url_path=TOKEN)
    updater.bot.set_webhook(HTTPS_URL + TOKEN)
    
    
    #method2: use polling 
    #updater.start_polling()
    
    
    
    updater.idle()



