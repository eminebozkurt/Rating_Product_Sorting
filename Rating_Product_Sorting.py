# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen puanları tarihe göre
# ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

import pandas as pd
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/Users/eminebozkurt/Desktop/vbo/Week4/hw2/amazon_review.csv")
df.head()
df.shape

# Adım 1: Ürünün ortalama puanını hesaplayınız.

df["overall"].mean() # 4.587589013224822

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
# • reviewTime değişkenini tarih değişkeni olarak tanıtmanız
# • reviewTime'ın max değerini current_date olarak kabul etmeniz
# • her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve gün cinsinden ifade edilen
# değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df.info()
df["reviewTime"] = df["reviewTime"].astype('datetime64[ns]')

df["reviewTime"].max()# Timestamp('2014-12-07 00:00:00')
current_date = pd.to_datetime('2014-12-7 0:0:0')
df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()

# df["days_cut"] = pd.qcut(df['days'], 4, labels=["1", "2", "3", "4"])
df["day_diff_cut"] = pd.qcut(df['day_diff'], 4, labels=["1", "2", "3", "4"])
df["day_diff_cut"].value_counts()
df["day_diff_cut"].value_counts().sum()
1236 / 4915 * 100
1228 / 4915 * 100
1223 / 4915 * 100

# 1 -> 25 -> 26
# 4 -> 24.9 -> 25
# 3 -> 24.9 -> 25
# 2 -> 24.8 -> 24



df.loc[df["days"] <= 30, "overall"].mean() * 26/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "overall"].mean() * 25/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "overall"].mean() * 25/100 + \
    df.loc[(df["days"] > 180), "overall"].mean() * 24/100 # 4.697477373833667

def time_based_weighted_average(dataframe, w1=26, w2=25, w3=25, w4=24):
    return dataframe.loc[df["days"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)


def time_based_weighted_average(dataframe, w1=26, w2=25, w3=25, w4=24):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100
time_based_weighted_average(df) #4.626870504061417



# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

df.head()
df.groupby("day_diff")


df.groupby("day_diff_cut").agg({"day_diff": "mean"})

# day_diff_cut
# 1            174.23382
# 2            352.53966
# 3            514.58795
# 4            709.47557 uzun zamandır değerlendirme yapmayanların sayısı oldukça yüksek

# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
# Adım 1: helpful_no değişkenini üretiniz.
# • total_vote bir yoruma verilen toplam up-down sayısıdır.
# • up, helpful demektir. up: 1, down: 0 desek
# • Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# • Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[df["total_vote"]> 0]
df.head()

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
# • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff, score_average_rating
# ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# • score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.

score_pos_neg_diff
score_average_rating
wilson_lower_bound


def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)





def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)











