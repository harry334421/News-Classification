from pandas import DataFrame
import json

MAX_NEWS = 50000
one_cat_max = 3000
train = 0.8
dev = 0.1
test = 0.1

if __name__ == '__main__':
    file = open("/Users/yangxuhang/PycharmProjects/MicrosoftPTA/News_Category_Dataset_v2.json", 'r', encoding='utf-8')
    df = DataFrame(columns=('cat', 'headline'))  # 生成空的pandas表
    count = 0
    politics = 0
    entertain = 0

    for line in file.readlines():
        try:
            dic = json.loads(line)
            if (dic['category'] == 'POLITICS'):  # 控制政治类新闻不要太多
                politics += 1
                if (politics >= one_cat_max):
                    continue

            if (dic['category'] == 'ENTERTAINMENT'):  # 控制政治类新闻不要太多
                entertain += 1
                if (entertain >= one_cat_max):
                    continue
            df.loc[count] = [dic['category'], dic['headline']]
            count += 1

        except:
            print(count)

        if count >= MAX_NEWS:
            break

    with open("dataset/train.tsv", 'w') as tsvfile:
        tsvfile.write(df.loc[:(MAX_NEWS * train)].to_csv(sep='\t', index=False))

    with open("dataset/dev.tsv", 'w') as tsvfile:
        tsvfile.write(df.loc[(MAX_NEWS * train):(MAX_NEWS * (train + dev))].to_csv(sep='\t', index=False))

    with open("dataset/test.tsv", 'w') as tsvfile:
        tsvfile.write(
            df.loc[(MAX_NEWS * (train + dev)):(MAX_NEWS * (train + dev + test))].to_csv(sep='\t', index=False))

    a = df['cat'].value_counts()
    print(a)
